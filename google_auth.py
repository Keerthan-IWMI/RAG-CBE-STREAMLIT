import streamlit as st
import requests
from urllib.parse import urlencode
import time
import json
import os
import hashlib

# Color palette
PRIMARY_COLOR = "#0F766E"
SECONDARY_COLOR = "#06B6D4"
ACCENT_COLOR = "#10B981"
BACKGROUND_LIGHT = "#F0FDFA"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#475569"

TOKEN_STORAGE_FILE = ".streamlit_auth_tokens_user.json"
BROWSER_ID_FILE = ".streamlit_browser_id.txt"

# ==================== BROWSER IDENTIFIER ====================
def get_browser_identifier():
    """
    Generate and store a persistent browser identifier.
    This survives page refreshes and identifies the device/browser.
    """
    if os.path.exists(BROWSER_ID_FILE):
        try:
            with open(BROWSER_ID_FILE, "r") as f:
                return f.read().strip()
        except:
            pass
    
    # Generate new browser ID if not exists
    import uuid
    browser_id = str(uuid.uuid4())
    try:
        with open(BROWSER_ID_FILE, "w") as f:
            f.write(browser_id)
    except:
        pass
    
    return browser_id

# ==================== GOOGLE OAUTH CLASS ====================
class GoogleOAuth:
    def __init__(self):
        self.client_id = st.secrets["client_id"]
        self.client_secret = st.secrets["client_secret"]
        self.redirect_uri = st.secrets.get("redirect_uri", "http://localhost:8501/oauth2callback")
        
        self.scope = (
            "https://www.googleapis.com/auth/userinfo.email "
            "https://www.googleapis.com/auth/userinfo.profile"
        )
        
    def get_authorization_url(self):
        params = {
            "client_id": self.client_id,
            "scope": self.scope,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "redirect_uri": self.redirect_uri,
        }
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    
    def get_tokens(self, code):
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        }
        
        r = requests.post(token_url, data=data)
        if r.status_code == 200:
            tokens = r.json()
            if 'expires_in' in tokens:
                tokens['expires_at'] = time.time() + tokens['expires_in']
            else:
                tokens['expires_at'] = time.time() + (2 * 60 * 60)
            return tokens
        
        st.error(f"Failed to get tokens: {r.text}")
        return None
    
    def get_user_info(self, access_token):
        userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(userinfo_url, headers=headers)
        if r.status_code == 200:
            return r.json()
        st.error(f"Failed to get user info: {r.text}")
        return None
    
    def refresh_access_token(self, refresh_token):
        """Refresh the access token using refresh token"""
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        r = requests.post(token_url, data=data)
        if r.status_code == 200:
            tokens = r.json()
            tokens['refresh_token'] = refresh_token
            if 'expires_in' in tokens:
                tokens['expires_at'] = time.time() + tokens['expires_in']
            return tokens
        return None

# ==================== USER-SPECIFIC TOKEN STORAGE ====================

def get_user_token_file(user_email, browser_id):
    """
    ✅ CREATE SEPARATE FILE FOR EACH USER + BROWSER COMBINATION!
    Format: .streamlit_tokens_{browser_hash}_{user_hash}.json
    """
    browser_hash = hashlib.md5(browser_id.encode()).hexdigest()[:8]
    user_hash = hashlib.md5(user_email.encode()).hexdigest()[:8]
    return f".streamlit_tokens_{browser_hash}_{user_hash}.json"

def get_all_user_token_files(browser_id):
    """Get all token files for current browser"""
    browser_hash = hashlib.md5(browser_id.encode()).hexdigest()[:8]
    token_files = []
    
    for file in os.listdir("."):
        if file.startswith(f".streamlit_tokens_{browser_hash}_") and file.endswith(".json"):
            token_files.append(file)
    
    return token_files

def save_tokens_to_file(tokens, user_info, browser_id):
    """✅ Save tokens to USER-SPECIFIC + BROWSER-SPECIFIC file"""
    try:
        user_email = user_info.get('email')
        if not user_email:
            print("❌ No email found in user info")
            return False
        
        # ✅ CRITICAL: Use user-specific + browser-specific file
        file_path = get_user_token_file(user_email, browser_id)
        
        user_data = {
            "tokens": {
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "expires_at": tokens.get("expires_at"),
                "expires_in": tokens.get("expires_in"),
            },
            "user_info": user_info,
            "browser_id": browser_id,
            "user_email": user_email,
            "is_active": True,  # ✅ Track active status
            "saved_at": time.time(),
            "last_accessed": time.time()
        }
        
        with open(file_path, "w") as f:
            json.dump(user_data, f, indent=2)
        
        print(f"✅ Saved tokens for user: {user_email}")
        print(f"   📁 Stored in: {file_path}")
        print(f"   🖥️ Browser ID: {browser_id[:8]}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving tokens: {e}")
        return False

def load_tokens_from_file(browser_id, user_email=None):
    """✅ Load tokens from specific user file or find active session"""
    try:
        # If specific user email provided
        if user_email:
            file_path = get_user_token_file(user_email, browser_id)
            if os.path.exists(file_path):
                return load_single_token_file(file_path)
            return None
        
        # Otherwise, find active session among all users for this browser
        token_files = get_all_user_token_files(browser_id)
        active_sessions = []
        
        for file_path in token_files:
            user_data = load_single_token_file(file_path)
            if user_data and user_data.get("is_active", False):
                active_sessions.append(user_data)
        
        # Return the most recently accessed active session
        if active_sessions:
            active_sessions.sort(key=lambda x: x.get("last_accessed", 0), reverse=True)
            return active_sessions[0]
        
        return None
    
    except Exception as e:
        print(f"❌ Error loading tokens: {e}")
        return None

def load_single_token_file(file_path):
    """Load and validate a single token file"""
    try:
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r") as f:
            user_data = json.load(f)
        
        tokens = user_data.get("tokens")
        user_info = user_data.get("user_info")
        
        # ✅ Validate session and refresh if needed
        if validate_user_session(tokens, user_info, file_path):
            # Update last accessed time
            user_data["last_accessed"] = time.time()
            with open(file_path, "w") as f:
                json.dump(user_data, f, indent=2)
            return user_data
        else:
            # Mark as inactive instead of deleting
            user_data["is_active"] = False
            with open(file_path, "w") as f:
                json.dump(user_data, f, indent=2)
            print(f"🔴 Marked session as inactive: {file_path}")
        
        return None
    
    except:
        return None

def validate_user_session(tokens, user_info, file_path):
    """Validate if user session is still valid - includes token refresh"""
    if not tokens or not user_info:
        return False
    
    if not isinstance(tokens, dict) or not isinstance(user_info, dict):
        return False
    
    expires_at = tokens.get("expires_at")
    current_time = time.time()
    
    # ✅ Check if token is expired or about to expire (within 5 minutes)
    if expires_at and (current_time > expires_at - 300):
        user_email = user_info.get("email", "Unknown")
        print(f"⏰ Token expired/expiring for {user_email}, attempting refresh...")
        google_oauth = GoogleOAuth()
        refresh_token = tokens.get("refresh_token")
        
        if refresh_token:
            new_tokens = google_oauth.refresh_access_token(refresh_token)
            if new_tokens:
                print(f"✅ Token refreshed for {user_email}")
                # ✅ Update tokens
                tokens['access_token'] = new_tokens.get('access_token')
                tokens['expires_at'] = new_tokens.get('expires_at')
                tokens['refresh_token'] = refresh_token
                return True
        
        print(f"❌ Failed to refresh token for {user_email}")
        return False
    
    return True

def mark_session_inactive(user_email, browser_id):
    """Mark a specific user session as inactive"""
    try:
        file_path = get_user_token_file(user_email, browser_id)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                user_data = json.load(f)
            
            user_data["is_active"] = False
            with open(file_path, "w") as f:
                json.dump(user_data, f, indent=2)
            
            print(f"🔴 Marked session as inactive: {user_email}")
        return True
    except Exception as e:
        print(f"❌ Error marking session inactive: {e}")
        return False

def delete_user_tokens(user_email, browser_id):
    """Delete stored tokens for specific user in THIS browser"""
    try:
        file_path = get_user_token_file(user_email, browser_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Deleted tokens for user: {user_email}")
        return True
    except Exception as e:
        print(f"❌ Error deleting tokens: {e}")
        return False

# ==================== AUTH FUNCTIONS ====================

def logout():
    """Enhanced logout - clear session and mark tokens as inactive for current user"""
    browser_id = get_browser_identifier()
    current_user = st.session_state.get("current_user_email")
    
    if current_user:
        mark_session_inactive(current_user, browser_id)
    
    keys_to_clear = [
        'google_authenticated', 'google_user', 'google_access_token',
        'google_refresh_token', 'session_start_time', 'token_expires_at',
        'current_user_email'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()

def restore_session_from_storage(browser_id):
    """
    ✅ Restore user session from USER-SPECIFIC + BROWSER-SPECIFIC file
    This finds the most recently active session for this browser
    """
    print(f"\n🔍 Attempting to restore session for browser: {browser_id[:8]}")
    
    stored_auth = load_tokens_from_file(browser_id)
    
    if stored_auth and stored_auth.get("is_active", False):
        tokens = stored_auth["tokens"]
        user_info = stored_auth["user_info"]
        user_email = user_info.get("email", "Unknown")
        
        print(f"✅ Successfully restored session for: {user_email}")
        
        # Restore to session state
        st.session_state.google_authenticated = True
        st.session_state.google_user = user_info
        st.session_state.google_access_token = tokens.get("access_token")
        st.session_state.session_start_time = time.time()
        st.session_state.current_user_email = user_email
        
        if 'refresh_token' in tokens:
            st.session_state.google_refresh_token = tokens['refresh_token']
        if 'expires_at' in tokens:
            st.session_state.token_expires_at = tokens['expires_at']
        
        return True
    
    print(f"   ❌ No valid active session found for this browser")
    return False

def check_google_auth():
    """Check if user is authenticated with Google - MULTI-USER FIXED VERSION"""
    google_oauth = GoogleOAuth()
    params = st.query_params
    browser_id = get_browser_identifier()

    print(f"\n{'='*60}")
    print(f"🔐 AUTH CHECK - Browser: {browser_id[:8]}")
    print(f"📊 Active users for this browser: {len(get_all_user_token_files(browser_id))}")
    print(f"{'='*60}")

    # ========== STEP 1: HANDLE OAUTH CALLBACK ==========
    if "code" in params:
        print(f"✅ OAuth callback detected - code received")
        code = params["code"]
        
        st.markdown(f"""
        <style>
        .auth-loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            background: linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #CCFBF1 100%);
            gap: 2rem;
        }}
        .spinner {{
            width: 60px;
            height: 60px;
            border: 4px solid #E0E7FF;
            border-top: 4px solid {PRIMARY_COLOR};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .auth-text {{
            font-size: 1.2rem;
            color: {PRIMARY_COLOR};
            font-weight: 600;
        }}
        </style>
        <div class="auth-loading">
            <div class="spinner"></div>
            <div class="auth-text">✨ Authenticating with Google...</div>
            <p style="color: {TEXT_SECONDARY};">Please wait while we verify your credentials</p>
        </div>
        """, unsafe_allow_html=True)
        
        tokens = google_oauth.get_tokens(code)
        
        if tokens and "access_token" in tokens:
            user_info = google_oauth.get_user_info(tokens["access_token"])
            
            if user_info:
                user_email = user_info.get('email')
                print(f"   ✅ User authenticated: {user_email}")
                
                st.session_state.google_authenticated = True
                st.session_state.google_user = user_info
                st.session_state.google_access_token = tokens["access_token"]
                st.session_state.session_start_time = time.time()
                st.session_state.current_user_email = user_email
                
                if 'refresh_token' in tokens:
                    st.session_state.google_refresh_token = tokens['refresh_token']
                if 'expires_at' in tokens:
                    st.session_state.token_expires_at = tokens['expires_at']
                
                # ✅ SAVE TO USER-SPECIFIC + BROWSER-SPECIFIC FILE
                save_tokens_to_file(tokens, user_info, browser_id)
                
                st.query_params.clear()
                st.rerun()
                return True
        
        st.error("❌ Authentication failed. Please try again.")
        return False

    # ========== STEP 2: CHECK SESSION STATE (FAST PATH) ==========
    if st.session_state.get("google_authenticated"):
        session_start = st.session_state.get("session_start_time")
        current_user = st.session_state.get("current_user_email")
        
        if session_start and (time.time() - session_start) < (2 * 60 * 60):
            print(f"✅ Valid in-memory session for: {current_user}")
            return True
        else:
            print(f"⏰ Session expired")
            logout()
            return False

    # ========== STEP 3: RESTORE FROM PERSISTENT STORAGE ✅ MULTI-USER FIXED ==========
    print(f"🔍 Checking persistent storage for this browser...")
    if restore_session_from_storage(browser_id):
        return True
    
    # ========== STEP 4: NO VALID AUTH FOUND ==========
    print(f"📝 No authentication found - showing login page\n")
    auth_url = google_oauth.get_authorization_url()
    show_login_page(auth_url)
    return False

def show_login_page(auth_url):
    """Display the beautiful login page"""
    st.markdown(
        f"""
        <style>
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes shimmer {{
            0% {{
                background-position: -1000px 0;
            }}
            100% {{
                background-position: 1000px 0;
            }}
        }}
        
        @keyframes float {{
            0%, 100% {{
                transform: translateY(0px);
            }}
            50% {{
                transform: translateY(-10px);
            }}
        }}
        
        .stApp {{
            background: linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #CCFBF1 100%);
        }}
        
        .login-card {{
            max-width: 420px;
            margin: 80px auto;
            padding: 50px 40px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(15, 118, 110, 0.1);
            text-align: center;
            animation: fadeInUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }}
        
        .login-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.1), transparent);
            animation: shimmer 3s infinite;
        }}
        
        .login-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(15, 118, 110, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .icon-wrapper {{
            display: inline-block;
            font-size: 48px;
            margin-bottom: 20px;
            animation: float 3s ease-in-out infinite;
        }}
        
        .login-title {{
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            animation: fadeInUp 0.6s ease-out 0.2s both;
        }}
        
        .login-subtitle {{
            color: {TEXT_SECONDARY};
            font-size: 16px;
            margin-bottom: 40px;
            animation: fadeInUp 0.6s ease-out 0.4s both;
        }}
        
        .google-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
            color: white;
            padding: 14px 36px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(15, 118, 110, 0.3);
            animation: fadeInUp 0.6s ease-out 0.6s both;
        }}
        
        .google-btn::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }}
        
        .google-btn:hover::before {{
            left: 100%;
        }}
        
        .google-btn:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 6px 25px rgba(15, 118, 110, 0.4);
        }}
        
        .google-btn:active {{
            transform: translateY(0) scale(0.98);
        }}
        
        .google-icon {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
        }}
        
        .security-badge {{
            margin-top: 30px;
            padding: 12px 20px;
            background: {BACKGROUND_LIGHT};
            border-radius: 10px;
            font-size: 13px;
            color: {TEXT_SECONDARY};
            animation: fadeInUp 0.6s ease-out 0.8s both;
        }}
        
        .security-icon {{
            color: {ACCENT_COLOR};
            margin-right: 6px;
        }}
        
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none !important;}}
        </style>
        <div class="login-card">
            <div class="icon-wrapper">🔐</div>
            <div class="login-title">CircularIQ</div>
            <div class="login-subtitle">CBE Decision Support Platform</div>
            <a href="{auth_url}" class="google-btn">
                <svg class="google-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path fill="#fff" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#fff" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#fff" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#fff" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Sign in with Google
            </a>
            <div class="security-badge">
                <span class="security-icon">🛡️</span>
                Secure OAuth 2.0 Authentication • Multi-User Support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )