"""
WSGI config for benchmark project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'benchmark.settings')

application = get_wsgi_application()

import django.core.handlers.wsgi
_application = django.core.handlers.wsgi.WSGIHandler()

def application(environ, start_response):
    if "BENCHMARK_DIR" in environ:
        os.environ["BENCHMARK_DIR"] = environ["BENCHMARK_DIR"]
    else:
        os.environ["BENCHMARK_DIR"] = "benchmark"
        os.environ["BENCHMARK_DIR"] = environ["BENCHMARK_DIR"]
    return _application(environ, start_response)
