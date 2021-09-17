#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_html_components as html
import sys
from pathlib import Path


# make available the repo folder as a package
sys.path.insert(0, Path(__file__).parent.parent.as_posix())


class ClusterDash(ComponentTemplate):

    _component_cls = html.Div

    def setup_layout(self, app):
        from ClusterDash import clusterDash  # noqa: F401

        self.child(app.layout)

        # app.layout = None  # the layout will available through self.layout.

        super().setup_layout(app)


extensions = [
    {
        'module': 'dasha.web.extensions.dasha',
        'config': {
            'template': ClusterDash,
            'title_text': 'Cluster Dash',
            }
        },
    {
        'module': 'dasha.web.extensions.cache',
        'config': {
            "CACHE_TYPE": 'redis',
            "CACHE_DEFAULT_TIMEOUT": 60 * 5,  # second
            "CACHE_KEY_PREFIX": 'tolteca_',
            "CACHE_REDIS_URL": f"redis://localhost:6379/0",
            }
        },
    ]
