"""Views for alignment browsers."""

import logging
import os

from pathlib import Path
from . import controllers
from flask import (
    current_app,
    Blueprint,
    Response,
    copy_current_request_context,
    render_template,
    request,
    session,
    abort
)
from flask_login import current_user, login_required

from app.bonsai import TokenObject, get_sample_by_id
from .partial import send_file_partial

alignviewers_bp = Blueprint(
    "alignviewers",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/alignviewers/static",
)

LOG = logging.getLogger(__name__)


@alignviewers_bp.route("/remote/static/", methods=["GET"])  # from case page
def remote_static():
    """Load large static files with special requirements."""
    base_path = Path(current_app.config["DATA_DIR"])
    file_path = base_path.joinpath(request.args.get("file"))
    if controllers.check_session_tracks(file_path) is False:
        abort(403)

    if not file_path.is_file():
        LOG.warning("file: %s cant be found", file_path)
        abort(404)

    if not os.access(file_path):
        LOG.warning("file: %s cant read by the system user", file_path)
        abort(500)

    range_header = request.headers.get("Range", None)
    if not range_header and (file_path.endswith(".bam") or file_path.endswith(".cram")):
        abort(500)
    
    new_resp = send_file_partial(file_path)
    return new_resp


@alignviewers_bp.route("/samples/<sample_id>/igv", methods=["GET"])  # from case page
@alignviewers_bp.route(
    "/samples/<sample_id>/<variant_id>/igv", methods=["GET"]
)  # from variants page
@login_required
def igv(sample_id, variant_id=None):
    """Visualize BAM alignments using igv.js (https://github.com/igvteam/igv.js)

    Args:
        sample(str): dislay_name of a case
        variant_id(str/None): variant _id or None
        chrom(str/None): requested chromosome [1-22], X, Y, [M-MT]
        start(int/None): start of the genomic interval to be displayed
        stop(int/None): stop of the genomic interval to be displayed

    Returns:
        a string, corresponging to the HTML rendering of the IGV alignments page
    """
    token = TokenObject(**current_user.get_id())
    sample_obj = get_sample_by_id(token, sample_id=sample_id)
    # make igv tracks to display
    display_obj = controllers.make_igv_tracks(sample_obj, variant_id, start=request.args.get("start"), stop=request.args.get("stop"))
    controllers.set_session_tracks(display_obj)

    response = Response(render_template("igv_viewer.html", **display_obj))

    @response.call_on_close
    @copy_current_request_context
    def clear_session_tracks():
        session.pop("igv_tracks", None)  # clean up igv session tracks

    return response