import streamlit as st
import requests
from urllib.parse import urlencode
import time
import json
import os

# Color palette - YOUR ORIGINAL COLORS
PRIMARY_COLOR = "#0F766E"      # Teal
SECONDARY_COLOR = "#06B6D4"    # Cyan
ACCENT_COLOR = "#10B981"       # Emerald
BACKGROUND_LIGHT = "#F0FDFA"   # Light teal
TEXT_PRIMARY = "#0F172A"       # Slate
TEXT_SECONDARY = "#475569"     # Slate gray

# File to store tokens persistently for ALL users
TOKEN_STORAGE_FILE = ".streamlit_auth_tokens_user.json"

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

# ==================== MULTI-USER STORAGE FUNCTIONS ====================

def save_tokens_to_file(tokens, user_info):
    """Save tokens and user info to JSON file using email as key"""
    try:
        user_email = user_info.get('email')
        if not user_email:
            print("❌ No email found in user info")
            return False
        
        # Load existing data with robust error handling
        all_users_data = {}
        if os.path.exists(TOKEN_STORAGE_FILE):
            try:
                with open(TOKEN_STORAGE_FILE, "r") as f:
                    all_users_data = json.load(f)
                # Validate structure is dict
                if not isinstance(all_users_data, dict):
                    print("⚠️ Invalid JSON structure, resetting...")
                    all_users_data = {}
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️ Corrupted JSON file, resetting: {e}")
                all_users_data = {}
        
        # Prepare user data
        user_data = {
            "tokens": tokens,
            "user_info": user_info,
            "saved_at": time.time(),
            "last_accessed": time.time()
        }
        
        # Update or add user data
        all_users_data[user_email] = user_data
        
        # Save back to file
        with open(TOKEN_STORAGE_FILE, "w") as f:
            json.dump(all_users_data, f, indent=2)
        
        print(f"✅ Saved tokens for user: {user_email}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving tokens: {e}")
        return False

def load_tokens_from_file(user_email):
    """Load tokens for specific user from multi-user JSON file"""
    try:
        if not user_email:
            return None
            
        if not os.path.exists(TOKEN_STORAGE_FILE):
            return None
        
        # Load with error handling for corrupted files
        try:
            with open(TOKEN_STORAGE_FILE, "r") as f:
                all_users_data = json.load(f)
            
            # Validate structure
            if not isinstance(all_users_data, dict):
                print("⚠️ Invalid JSON structure in token file")
                return None
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Corrupted token file: {e}")
            return None
        
        # Load specific user
        if user_email in all_users_data:
            user_data = all_users_data[user_email]
            tokens = user_data.get("tokens")
            user_info = user_data.get("user_info")
            
            if validate_user_session(tokens, user_info):
                # Update last accessed time
                user_data["last_accessed"] = time.time()
                save_all_users_data(all_users_data)
                return {"tokens": tokens, "user_info": user_info}
            else:
                # Remove expired/invalid session
                del all_users_data[user_email]
                save_all_users_data(all_users_data)
        
        return None
    
    except Exception as e:
        print(f"❌ Error loading tokens: {e}")
        return None

def validate_user_session(tokens, user_info):
    """Validate if user session is still valid"""
    if not tokens or not user_info:
        return False
    
    # Check if tokens and user_info are proper dictionaries
    if not isinstance(tokens, dict) or not isinstance(user_info, dict):
        return False
    
    # Check token expiration
    expires_at = tokens.get("expires_at")
    if expires_at and time.time() > expires_at:
        # Try to refresh token
        google_oauth = GoogleOAuth()
        refresh_token = tokens.get("refresh_token")
        
        if refresh_token:
            new_tokens = google_oauth.refresh_access_token(refresh_token)
            if new_tokens:
                new_tokens['refresh_token'] = refresh_token
                save_tokens_to_file(new_tokens, user_info)
                return True
        return False
    
    return True

def save_all_users_data(all_users_data):
    """Helper function to save all users data"""
    try:
        with open(TOKEN_STORAGE_FILE, "w") as f:
            json.dump(all_users_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving all users data: {e}")
        return False

def delete_tokens_from_file(user_email=None):
    """Delete stored tokens for specific user or current user"""
    try:
        if not os.path.exists(TOKEN_STORAGE_FILE):
            return True
            
        # Load with error handling
        try:
            with open(TOKEN_STORAGE_FILE, "r") as f:
                all_users_data = json.load(f)
            
            if not isinstance(all_users_data, dict):
                print("⚠️ Invalid structure in delete operation")
                return False
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Corrupted file during delete: {e}")
            return False
            
        # If no user_email provided, use current session user
        if not user_email and st.session_state.get('google_user'):
            user_email = st.session_state.google_user.get('email')
        
        if not user_email:
            return False
        
        # Delete specific user
        if user_email in all_users_data:
            del all_users_data[user_email]
            with open(TOKEN_STORAGE_FILE, "w") as f:
                json.dump(all_users_data, f, indent=2)
            print(f"✅ Deleted tokens for user: {user_email}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error deleting tokens: {e}")
        return False

def get_all_stored_users():
    """Get list of all users with stored sessions (for admin/debug purposes)"""
    try:
        if not os.path.exists(TOKEN_STORAGE_FILE):
            return []
        
        # Load with error handling
        try:    
            with open(TOKEN_STORAGE_FILE, "r") as f:
                all_users_data = json.load(f)
            
            if not isinstance(all_users_data, dict):
                return []
                
        except (json.JSONDecodeError, Exception):
            return []
        
        return list(all_users_data.keys())
    except Exception as e:
        print(f"Error getting stored users: {e}")
        return []

# ==================== UPDATED AUTH FUNCTIONS ====================

def logout():
    """Enhanced logout - clear session and remove user from storage"""
    user_email = None
    if st.session_state.get('google_user'):
        user_email = st.session_state.google_user.get('email')
    
    keys_to_clear = [
        'google_authenticated', 'google_user', 'google_access_token',
        'google_refresh_token', 'session_start_time', 'token_expires_at',
        'current_user_email'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Remove from storage
    if user_email:
        delete_tokens_from_file(user_email)
    
    st.rerun()

def check_google_auth():
    """Check if user is authenticated with Google - with multi-user storage"""
    google_oauth = GoogleOAuth()
    params = st.query_params

    print(f"🔍 DEBUG - Auth Check Started")
    print(f"   redirect_uri: {google_oauth.redirect_uri}")

    # ========== HANDLE OAUTH CALLBACK ==========
    if "code" in params:
        print(f"✅ OAuth callback detected")
        code = params["code"]
        
        # Show loading screen
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
        
        # Get tokens
        tokens = google_oauth.get_tokens(code)
        
        if tokens and "access_token" in tokens:
            user_info = google_oauth.get_user_info(tokens["access_token"])
            
            if user_info:
                user_email = user_info.get('email')
                print(f"✅ User authenticated: {user_email}")
                
                # Store in session state
                st.session_state.google_authenticated = True
                st.session_state.google_user = user_info
                st.session_state.google_access_token = tokens["access_token"]
                st.session_state.session_start_time = time.time()
                st.session_state.current_user_email = user_email
                
                if 'refresh_token' in tokens:
                    st.session_state.google_refresh_token = tokens['refresh_token']
                if 'expires_at' in tokens:
                    st.session_state.token_expires_at = tokens['expires_at']
                
                # Save to multi-user storage
                save_tokens_to_file(tokens, user_info)
                
                # Clear query params and redirect
                st.query_params.clear()
                st.rerun()
                return True
        
        st.error("❌ Authentication failed. Please try again.")
        return False

    # ========== CHECK SESSION STATE ==========
    if st.session_state.get("google_authenticated"):
        current_user_email = st.session_state.get("current_user_email")
        session_start = st.session_state.get("session_start_time")
        
        if session_start and (time.time() - session_start) < (2 * 60 * 60):
            print(f"✅ Valid session for: {current_user_email}")
            return True
        else:
            print(f"⏰ Session expired - logging out")
            logout()
            return False

    # ========== CHECK PERSISTENT STORAGE ==========
    print(f"🔍 Checking persistent storage...")
    
    # 🎯 FIXED: Check if current user has valid tokens in storage
    current_user_email = st.session_state.get("current_user_email")
    
    if current_user_email:
        # Check if this specific user has valid tokens in storage
        stored_auth = load_tokens_from_file(current_user_email)
        if stored_auth:
            tokens = stored_auth["tokens"]
            user_info = stored_auth["user_info"]
            
            print(f"✅ Restored session for: {current_user_email}")
            
            # Restore session state
            st.session_state.google_authenticated = True
            st.session_state.google_user = user_info
            st.session_state.google_access_token = tokens["access_token"]
            st.session_state.session_start_time = time.time()
            st.session_state.current_user_email = current_user_email
            
            if 'refresh_token' in tokens:
                st.session_state.google_refresh_token = tokens['refresh_token']
            if 'expires_at' in tokens:
                st.session_state.token_expires_at = tokens['expires_at']
            
            return True

    # ========== SHOW LOGIN PAGE ==========
    print(f"📝 No authentication found - showing login page")
    auth_url = google_oauth.get_authorization_url()
    show_login_page(auth_url)
    return False

def show_login_page(auth_url):
    """Display the beautiful login page - WITH YOUR ORIGINAL DESIGN"""
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
