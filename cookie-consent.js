// Google Consent Mode v2 Implementation
// Initialize consent mode with default denied state (for UK/EU compliance)
window.dataLayer = window.dataLayer || [];
function gtag() { dataLayer.push(arguments); }

// Set default consent to 'denied' before any tracking
gtag('consent', 'default', {
  'analytics_storage': 'denied',
  'ad_storage': 'denied',
  'ad_user_data': 'denied',
  'ad_personalization': 'denied',
  'wait_for_update': 500
});

// Cookie consent banner functionality
(function() {
  'use strict';

  const COOKIE_NAME = 'cookie_consent';
  const COOKIE_EXPIRY_DAYS = 365;

  // Check if consent has already been given
  function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
  }

  // Set cookie
  function setCookie(name, value, days) {
    const date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    const expires = `expires=${date.toUTCString()}`;
    document.cookie = `${name}=${value};${expires};path=/;SameSite=Lax`;
  }

  // Update consent mode
  function updateConsentMode(analyticsGranted) {
    gtag('consent', 'update', {
      'analytics_storage': analyticsGranted ? 'granted' : 'denied'
    });
  }

  // Handle accept
  function acceptCookies() {
    setCookie(COOKIE_NAME, 'accepted', COOKIE_EXPIRY_DAYS);
    updateConsentMode(true);
    hideBanner();
  }

  // Handle reject
  function rejectCookies() {
    setCookie(COOKIE_NAME, 'rejected', COOKIE_EXPIRY_DAYS);
    updateConsentMode(false);
    hideBanner();
  }

  // Hide banner
  function hideBanner() {
    const banner = document.getElementById('cookie-consent-banner');
    if (banner) {
      banner.style.display = 'none';
    }
  }

  // Show banner
  function showBanner() {
    const banner = document.createElement('div');
    banner.id = 'cookie-consent-banner';
    banner.className = 'cookie-consent-banner';
    banner.innerHTML = `
      <div class="cookie-consent-content">
        <div class="cookie-consent-text">
          <p><strong>We use cookies</strong></p>
          <p>This site uses Google Analytics cookies to help us understand how visitors use the site. These cookies collect information in an anonymous form. You can choose to accept or reject analytics cookies.</p>
        </div>
        <div class="cookie-consent-buttons">
          <button id="cookie-accept" class="cookie-btn cookie-btn-accept">Accept Analytics</button>
          <button id="cookie-reject" class="cookie-btn cookie-btn-reject">Reject</button>
        </div>
      </div>
    `;

    document.body.appendChild(banner);

    // Add event listeners
    document.getElementById('cookie-accept').addEventListener('click', acceptCookies);
    document.getElementById('cookie-reject').addEventListener('click', rejectCookies);
  }

  // Initialize on page load
  function init() {
    const consent = getCookie(COOKIE_NAME);

    if (consent === 'accepted') {
      // User previously accepted
      updateConsentMode(true);
    } else if (consent === 'rejected') {
      // User previously rejected
      updateConsentMode(false);
    } else {
      // No consent yet, show banner
      showBanner();
    }
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
