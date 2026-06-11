# ============================================================
# auth.py — Authentication, Session Management, Access Logging
# ============================================================

import os
import json
import hashlib
import csv
import streamlit as st
from datetime import datetime
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────
_SRC      = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_SRC)
USERS_FILE = os.path.join(_ROOT, "users.json")
LOG_DIR    = os.path.join(_ROOT, "logs")
LOG_FILE   = os.path.join(LOG_DIR, "access_log.csv")

LOG_HEADERS = ["timestamp", "username", "role", "event", "detail", "ip"]


# ══════════════════════════════════════════════════════════════
# Password Hashing  (PBKDF2-SHA256, no external deps)
# ══════════════════════════════════════════════════════════════

def _hash_password(password: str) -> str:
    """Hash a plain-text password. Returns  salt:key  string."""
    salt = os.urandom(32).hex()
    key  = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 260_000
    )
    return f"{salt}:{key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plain-text password against a stored  salt:key  hash."""
    try:
        salt, key_hex = stored_hash.split(":", 1)
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 260_000
        )
        return key.hex() == key_hex
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# User Store  (users.json)
# ══════════════════════════════════════════════════════════════

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def get_all_users() -> list[dict]:
    """Return list of user dicts (without password hashes)."""
    users = _load_users()
    return [
        {
            "username":   uname,
            "role":       data.get("role", "user"),
            "full_name":  data.get("full_name", ""),
            "created_at": data.get("created_at", ""),
            "created_by": data.get("created_by", ""),
        }
        for uname, data in users.items()
    ]


def add_user(username: str, password: str, role: str,
             full_name: str, created_by: str) -> tuple[bool, str]:
    """
    Add a new user. Returns (success, message).
    """
    if not username.strip():
        return False, "Username cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if role not in ("admin", "user"):
        return False, "Role must be 'admin' or 'user'."

    users = _load_users()
    if username in users:
        return False, f"User '{username}' already exists."

    users[username] = {
        "password_hash": _hash_password(password),
        "role":          role,
        "full_name":     full_name.strip(),
        "created_at":    str(datetime.now().date()),
        "created_by":    created_by,
    }
    _save_users(users)
    log_event(created_by, "admin", "USER_CREATED", f"created user: {username} (role={role})")
    return True, f"User '{username}' created successfully."


def delete_user(username: str, deleted_by: str) -> tuple[bool, str]:
    """Delete a user. Admins cannot delete themselves."""
    if username == deleted_by:
        return False, "You cannot delete your own account."
    users = _load_users()
    if username not in users:
        return False, f"User '{username}' not found."
    del users[username]
    _save_users(users)
    log_event(deleted_by, "admin", "USER_DELETED", f"deleted user: {username}")
    return True, f"User '{username}' deleted."


def change_password(username: str, new_password: str,
                    changed_by: str) -> tuple[bool, str]:
    if len(new_password) < 4:
        return False, "Password must be at least 4 characters."
    users = _load_users()
    if username not in users:
        return False, "User not found."
    users[username]["password_hash"] = _hash_password(new_password)
    _save_users(users)
    log_event(changed_by, "admin", "PASSWORD_CHANGED", f"changed password for: {username}")
    return True, "Password updated successfully."


# ══════════════════════════════════════════════════════════════
# Access Log
# ══════════════════════════════════════════════════════════════

def _ensure_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)


def log_event(username: str, role: str, event: str, detail: str = ""):
    """Append one row to the access log."""
    # Strip emojis from detail to keep the log file ASCII-safe
    detail_clean = detail.encode("ascii", errors="ignore").decode("ascii")
    _ensure_log()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username, role, event, detail_clean, "localhost"
        ])


def get_access_log() -> list[dict]:
    """Return all log rows as a list of dicts."""
    _ensure_log()
    import pandas as pd
    try:
        df = pd.read_csv(LOG_FILE, encoding="utf-8")
        return df.to_dict("records")
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# Session Management  (Streamlit session_state)
# ══════════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        "logged_in": False,
        "username":  "",
        "role":      "",
        "full_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_logged_in() -> bool:
    _init_session()
    return st.session_state.get("logged_in", False)


def current_user() -> dict:
    return {
        "username":  st.session_state.get("username", ""),
        "role":      st.session_state.get("role", ""),
        "full_name": st.session_state.get("full_name", ""),
    }


def is_admin() -> bool:
    return st.session_state.get("role") == "admin"


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Validate credentials and set session state.
    Returns (success, message).
    """
    _init_session()
    users = _load_users()

    if username not in users:
        log_event(username, "—", "LOGIN_FAILED", "username not found")
        return False, "Invalid username or password."

    user = users[username]
    if not _verify_password(password, user["password_hash"]):
        log_event(username, "—", "LOGIN_FAILED", "wrong password")
        return False, "Invalid username or password."

    # Success — set session
    st.session_state["logged_in"] = True
    st.session_state["username"]  = username
    st.session_state["role"]      = user["role"]
    st.session_state["full_name"] = user.get("full_name", username)

    log_event(username, user["role"], "LOGIN_SUCCESS")
    return True, "Login successful."


def logout():
    user = current_user()
    log_event(user["username"], user["role"], "LOGOUT")
    for k in ["logged_in", "username", "role", "full_name"]:
        st.session_state[k] = "" if k != "logged_in" else False
    st.rerun()


# ══════════════════════════════════════════════════════════════
# Login Page UI  (rendered by app.py when not logged in)
# ══════════════════════════════════════════════════════════════

def render_login_page():
    """Render the full-screen login form."""
    # Centre the login card
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
<div style='text-align:center; margin-bottom:24px'>
    <div style='font-size:3rem'>📡</div>
    <h2 style='margin:6px 0 2px'>Telecom Churn Analytics</h2>
    <p style='color:#888; font-size:0.9rem'>Sign in to continue</p>
</div>
""", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### 🔐 Login")
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Sign In", type="primary", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    success, msg = login(username, password)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("""
<p style='text-align:center; color:#aaa; font-size:0.8rem; margin-top:20px'>
    Divyesh Joshi | MCA-II | IIMS Chinchwad, Pune
</p>
""", unsafe_allow_html=True)
