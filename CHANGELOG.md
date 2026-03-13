# Changelog

## [2026-03-13] — Themes, Hotkey & UI Polish
### Added
- Light and Dark mode toggle in Settings → Appearance
- Theme applies live — no restart needed
- Settings window now also reflects the chosen theme
- Hotkey status indicator in Settings (shows ✓ Active or error message)
- Transcription sensitivity description now matches Strict/Permissive labels

### Fixed
- Global hotkey (show/hide overlay) now works correctly when launched via pythonw
- Session briefing placeholder no longer shows confusing "Claude" reference
- Settings window was always dark regardless of selected theme

---

## [2026-03-13] — Batch 1 Features
### Added
- **Clipboard copy** — one-click copy button on every suggestion
- **Global hotkey** — show/hide the overlay without touching the mouse (default: Ctrl+Shift+Space, configurable in Settings)
- **Session export** — export the full transcript and suggestions as a `.txt` file at the end of a session

---

## [2026-03-13] — Auto-Update System
### Added
- Background version check on every launch
- Update banner appears automatically when a new version is available
- "Update Now" button downloads and installs the update automatically
- "Restart" button appears after a successful update
- Updates install via ZIP download — no Git required
- Dependencies installed automatically as part of each update

---

## [2026-03-13] — Foundation
### Added
- Opacity slider defaults to 100% on launch
- CMD window no longer appears when launching via `python main.py` on Windows
- Stealth mode — hides overlay from screen sharing/recording (Windows 10 2004+)
- Speaker labels — distinguishes Your voice vs Their voice in the transcript
- Pre-session briefing — paste context before starting (job description, agenda, etc.)
- Custom system prompts per mode in Settings
- Confidence threshold filter — Strict/Permissive slider controls transcription sensitivity
- Follow-up question tracker — surfaces things to revisit from the conversation
- Session history tab — browse previous suggestions from this session
