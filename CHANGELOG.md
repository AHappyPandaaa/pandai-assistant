# Changelog

## [2.0] — Complete V2 Overhaul

### Redesigned
- **Apple-inspired UI** — full dark/light design system with layered surfaces, smooth radius, and refined typography throughout
- **Transcript-first UX** — removed all conversation modes; the app now continuously transcribes and lets you drive analysis on your terms
- **On-demand analysis** — highlight any text in the transcript and click Analyze to get an immediate, actionable response; questions aimed at you get ready-to-say answers, topics get talking points
- **Session persistence** — every recording is saved as a named session with full transcript and all analyses; sessions survive restarts and appear in the History tab
- **Auto-titling** — sessions are automatically titled in the background after recording ends using the conversation content
- **Background highlighting** — key phrases are extracted every ~12 seconds and highlighted in the transcript to surface what matters

### Added
- **Delete sessions** — remove individual sessions from history; deletions are permanent and sync to disk
- **Connected topics** — each analysis surfaces adjacent concepts related to what was said
- **Follow-up chips** — suggested follow-up questions appear as clickable chips after each analysis
- **Briefing field** — prime the AI with context before a call (job description, agenda, names) to get more relevant analysis
- **Crash logging** — unhandled exceptions write a full traceback to the Desktop for easier diagnosis

### Fixed
- Analysis speed improved significantly by switching to Haiku for on-demand responses
- Session expand no longer crashes on long transcript text
- Window correctly positions on multi-monitor setups
- Crash log path now resolves correctly when Desktop is on OneDrive
- Background workers (highlight, title) no longer cause mid-session segfaults
