from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render


def signup(request):
    """회원가입: Django 기본 User와 기본 폼(아이디 중복·비밀번호 규칙 검사 포함)을 사용한다."""
    if request.user.is_authenticated:
        return redirect("analyses:index")
    form = UserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("analyses:index")
    return render(request, "accounts/signup.html", {"form": form})
