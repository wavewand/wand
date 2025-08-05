"""
🌐 Productivity & Workflow Integrations

Communication, storage, and documentation tools for Wand
"""

from .communication import (
    CalendarIntegration,
    DiscordIntegration,
    EmailIntegration,
    NotionIntegration,
    TelegramIntegration,
)
from .documentation import ConfluenceIntegration, GitBookIntegration, MarkdownIntegration, PDFIntegration
from .file_storage import DropboxIntegration, FTPIntegration, GoogleDriveIntegration, OneDriveIntegration, S3Integration

# Initialize integration instances
discord_integration = DiscordIntegration()
telegram_integration = TelegramIntegration()
email_integration = EmailIntegration()
calendar_integration = CalendarIntegration()
notion_integration = NotionIntegration()

gdrive_integration = GoogleDriveIntegration()
dropbox_integration = DropboxIntegration()
onedrive_integration = OneDriveIntegration()
s3_integration = S3Integration()
ftp_integration = FTPIntegration()

confluence_integration = ConfluenceIntegration()
gitbook_integration = GitBookIntegration()
markdown_integration = MarkdownIntegration()
pdf_integration = PDFIntegration()

__all__ = [
    # Communication
    "discord_integration",
    "telegram_integration",
    "email_integration",
    "calendar_integration",
    "notion_integration",
    # File storage
    "gdrive_integration",
    "dropbox_integration",
    "onedrive_integration",
    "s3_integration",
    "ftp_integration",
    # Documentation
    "confluence_integration",
    "gitbook_integration",
    "markdown_integration",
    "pdf_integration",
]
