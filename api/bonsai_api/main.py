"""Main entrypoint for API server."""

from logging.config import dictConfig

from fastapi import FastAPI

from .config import settings
from .db.utils import close_mongo_connection, connect_to_mongo
from .extensions.ldap_extension import ldap_connection
from .internal.middlewares import configure_cors
from .routers import (
    auth,
    cluster,
    export,
    groups,
    jobs,
    locations,
    resources,
    root,
    samples,
    users,
)

dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "root": {"level": "DEBUG", "handlers": ["wsgi"]},
    }
)

app = FastAPI(title="Bonsai")

# configure CORS
configure_cors(app)

# configure events
app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

if settings.use_ldap_auth:
    app.add_event_handler("startup", ldap_connection.init_app)
    app.add_event_handler("shutdown", ldap_connection.teardown)


# add api routes
app.include_router(root.router)
app.include_router(users.router)
app.include_router(samples.router)
app.include_router(groups.router)
app.include_router(locations.router)
app.include_router(cluster.router)
app.include_router(export.router)
app.include_router(resources.router)
app.include_router(auth.router)
app.include_router(jobs.router)
