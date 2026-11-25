"""Django 설정. 비밀값과 주소는 .env(환경변수)에서 읽는다."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env(name, default=None):
    return os.environ.get(name, default)


def env_bool(name, default=False):
    return str(env(name, str(default))).lower() in ("1", "true", "yes", "on")


def env_list(name, default=""):
    return [v.strip() for v in env(name, default).split(",") if v.strip()]


SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "accounts",
    "analyses",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ── DB: 기본은 PostgreSQL. USE_SQLITE=1이면 빠른 로컬 테스트용 SQLite ──
if env_bool("USE_SQLITE", False):
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("POSTGRES_DB", "vision_action"),
            "USER": env("POSTGRES_USER", "vision"),
            "PASSWORD": env("POSTGRES_PASSWORD", ""),
            "HOST": env("POSTGRES_HOST", "localhost"),
            "PORT": env("POSTGRES_PORT", "5432"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

# 업로드 영상: DB에는 경로만, 파일은 MEDIA_ROOT(Compose에서는 django·celery 공유 볼륨)
MEDIA_URL = "media/"
MEDIA_ROOT = Path(env("MEDIA_ROOT", str(BASE_DIR / "media")))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── 로그인 ──
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "analyses:index"
LOGOUT_REDIRECT_URL = "login"

# ── DRF: 세션 로그인 사용자만 API 사용 ──
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",  # curl 검증용
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# ── Celery / Redis ──
REDIS_URL = env("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", False)  # 테스트용

# ── AI 서비스 / 입력 제한 ──
AI_SERVICE_URL = env("AI_SERVICE_URL", "http://localhost:8001")
AI_REQUEST_TIMEOUT = float(env("AI_REQUEST_TIMEOUT", "300"))
MAX_UPLOAD_MB = int(env("MAX_UPLOAD_MB", "100"))
ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"]

# IP Webcam: 허용 목록에 있는 카메라만. 형식 "이름|URL,이름|URL"
ALLOWED_CAMERAS = {}
for item in env_list("ALLOWED_CAMERAS", ""):
    name, _, url = item.partition("|")
    if name and url:
        ALLOWED_CAMERAS[name.strip()] = url.strip()
CAMERA_CAPTURE_SECONDS = float(env("CAMERA_CAPTURE_SECONDS", "8"))
CAMERA_CONNECT_TIMEOUT = float(env("CAMERA_CONNECT_TIMEOUT", "10"))

# 화면에 '판단 불확실'로 표시할 기준(FastAPI의 uncertain과 같은 값)
UNCERTAIN_BELOW = float(env("UNCERTAIN_BELOW", "0.3"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
