from django.contrib.auth.models import User
from django.test import TestCase


class AccountTests(TestCase):
    def test_signup_login_logout(self):
        res = self.client.post("/accounts/signup/", {"username": "newbie", "password1": "Strong-pass-2026", "password2": "Strong-pass-2026"})
        self.assertRedirects(res, "/analyses/")
        self.assertTrue(User.objects.filter(username="newbie").exists())
        self.client.post("/accounts/logout/")
        self.assertEqual(self.client.get("/analyses/").status_code, 302)
        res = self.client.post("/accounts/login/", {"username": "newbie", "password": "Strong-pass-2026"})
        self.assertRedirects(res, "/analyses/")

    def test_signup_rejects_duplicate_and_weak_password(self):
        User.objects.create_user("taken", password="whatever-123")
        res = self.client.post("/accounts/signup/", {"username": "taken", "password1": "Strong-pass-2026", "password2": "Strong-pass-2026"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.context["form"].errors.get("username"))
        res = self.client.post("/accounts/signup/", {"username": "weak", "password1": "1234", "password2": "1234"})
        self.assertTrue(res.context["form"].errors.get("password2"))
        self.assertFalse(User.objects.filter(username="weak").exists())
