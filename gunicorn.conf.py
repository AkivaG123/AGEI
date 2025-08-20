# gunicorn.conf.py
bind = "0.0.0.0:8080"
workers = 1
timeout = 300  # 5 minutes
worker_class = "sync"
max_requests = 1000
max_requests_jitter = 50
