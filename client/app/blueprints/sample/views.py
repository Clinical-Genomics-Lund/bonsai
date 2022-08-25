"""Declaration of views for samples"""
from flask import Blueprint, current_app, render_template
from app.mimer import cgmlst_cluster_samples
from flask_login import login_required

samples_bp = Blueprint(
    "samples", __name__, template_folder="templates", static_folder="static"
)


@samples_bp.route("/samples")
@login_required
def samples():
    """Samples view."""
    return render_template("samples.html")


@samples_bp.route("/samples/<sample_id>")
@login_required
def sample(sample_id):
    """Samples view."""
    return render_template("sample.html", sample_id=sample_id)


@samples_bp.route("/cluster/", methods=['GET', 'POST'])
@login_required
def cluster(sample_id):
    """Samples view."""
    token = TokenObject(**current_user.get_id())

    if request.method == 'POST':
        samples = request.body['samples']
        cgmlst_cluster_samples(token, samples=samples)
    return render_template("sample.html", sample_id=sample_id)
