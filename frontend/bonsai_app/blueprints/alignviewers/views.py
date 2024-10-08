"""Views for alignment browsers."""

import logging

from flask import (
    Blueprint,
    Response,
    copy_current_request_context,
    render_template,
    request,
    session,
)
from flask_login import current_user, login_required

from ...bonsai import TokenObject, get_sample_by_id
from . import controllers

alignviewers_bp = Blueprint(
    "alignviewers",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/alignviewers/static",
)

LOG = logging.getLogger(__name__)


@alignviewers_bp.route("/samples/<sample_id>/igv", methods=["GET"])  # from case page
@alignviewers_bp.route(
    "/samples/<sample_id>/<variant_id>/igv", methods=["GET"]
)  # from variants page
@login_required
def igv(sample_id: str, variant_id: str | None = None):
    """Visualize BAM alignments using igv.js (https://github.com/igvteam/igv.js)

    :param sample_id: _description_
    :type sample_id: str
    :param variant_id: _description_, defaults to None
    :type variant_id: str | None, optional
    :return: a string, corresponging to the HTML rendering of the IGV alignments page
    :rtype: str

    Args:
        start(int/None): start of the genomic interval to be displayed
        stop(int/None): stop of the genomic interval to be displayed
    """
    token = TokenObject(**current_user.get_id())
    sample_obj = get_sample_by_id(token, sample_id=sample_id)
    # make igv tracks to display
    display_obj = controllers.make_igv_tracks(
        sample_obj,
        variant_id,
        start=request.args.get("start"),
        stop=request.args.get("stop"),
    )
    controllers.set_session_tracks(display_obj)

    igv_config = display_obj.model_dump(mode="json", by_alias=True, exclude_none=True)
    response = Response(render_template("igv_viewer.html", igv_config=igv_config))

    @response.call_on_close
    @copy_current_request_context
    def clear_session_tracks():
        session.pop("igv_tracks", None)  # clean up igv session tracks

    return response
