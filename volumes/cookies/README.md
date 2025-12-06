# YouTube Cookies for yt-dlp

This directory is for storing YouTube cookies to allow yt-dlp to access age-restricted, member-only, or region-locked streams.

## How to Get YouTube Cookies

### Method 1: Using Browser Extension (Recommended)

1. **Install a cookie exporter extension:**
   - Chrome/Edge: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   - Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. **Export cookies:**
   - Open YouTube and make sure you're logged in
   - Navigate to the stream you want to access
   - Click the extension icon
   - Export cookies for `youtube.com`
   - Save as `youtube_cookies.txt`

3. **Place the file:**
   - Copy `youtube_cookies.txt` to this directory (`volumes/cookies/`)

4. **Configure environment:**
   - Edit `.env` file in the project root
   - Uncomment and set: `COOKIES_FILE=/cookies/youtube_cookies.txt`
   - Restart the stream-capture service: `docker compose up -d --build stream-capture`

### Method 2: Manual Export from Browser

You can also manually export cookies using browser developer tools, but the extension method is much easier.

## Usage

Once configured, yt-dlp will use these cookies to authenticate with YouTube, allowing access to:
- Age-restricted videos
- Member-only live streams
- Region-locked content
- Private/unlisted streams (if you have access)

## Security Notes

- **Never commit cookies to git!** The `.gitignore` already excludes this directory.
- Cookies contain your authentication data - keep them private
- Cookies expire periodically - you may need to re-export them
- If you see authentication errors, try exporting fresh cookies

## Troubleshooting

If cookies aren't working:
1. Make sure the file path in `.env` matches the actual file location
2. Verify cookies are in Netscape format (should start with `# Netscape HTTP Cookie File`)
3. Check that cookies haven't expired - export fresh ones
4. Check stream-capture logs: `docker logs flowium-stream-capture`
5. You should see: `🍪 Using cookies from: /cookies/youtube_cookies.txt`
