#!/bin/python3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import poplib
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr
import argparse
import json

DEFAULT_KEYS = json.load(open("mail_keys.json"))


class SmtpServer:
    def __init__(self, host: str, username: str, password: str):
        self.host = host
        self.username = username
        self.password = password
        self.smtp = smtplib.SMTP_SSL(self.host, 465)

    def send(
        self,
        receivers: list[str],
        title: str,
        content: str,
        attachments: list[str] | None = None,
    ):
        """Send an email

        Args:
            receivers (list[str]): receivers' email addresses
            title (str): title of the email
            content (str): content of the email
            attachments (list[str] | None, optional): attachments. Defaults to None.
        """
        self.smtp.connect(self.host, 465)
        self.smtp.login(self.username, self.password)
        message = MIMEMultipart()
        message["From"] = self.username
        message["To"] = ",".join(receivers)
        message["Subject"] = title
        message.attach(MIMEText(content, "plain", "utf-8"))
        if attachments is not None:
            for attachment in attachments:
                attach = MIMEText(open(attachment, "rb").read(), "base64", "utf-8")
                attach["Content-Type"] = "application/octet-stream"
                attach["Content-Disposition"] = (
                    'attachment; filename="%s"' % attachment.split("/")[-1]
                )
                message.attach(attach)
        self.smtp.sendmail(self.username, receivers, message.as_string())
        self.smtp.quit()

def send_default(title: str, content: str, attachments: list[str] | None = None):
    """Send an email using the default mail server

    Args:
        title (str): mail title
        content (str): mail content
        attachments (list[str] | None, optional): attachments. Defaults to None.
    """
    mail_server = SmtpServer(
        host=DEFAULT_KEYS["smtphost"],
        username=DEFAULT_KEYS["username"],
        password=DEFAULT_KEYS["password"],
    )
    mail_server.send(
        receivers=["2274289773@qq.com"],
        title=title,
        content=content,
        attachments=attachments,
    )


class POP3Server:
    def __init__(
        self, host: str, username: str, password: str, ignore_past_mails: bool = True
    ):
        """Create a POP3 server instance

        Args:
            host (str): host name
            username (str): username
            password (str): password
        """
        self.host = host
        self.username = username
        self.password = password
        self.lastid = None
        self.pop3 = None
        if ignore_past_mails:
            self.lastid = self.update_latest_id()
        
    def update(self):
        if self.pop3 is not None:
            self.pop3.quit()
        self.pop3 = poplib.POP3_SSL(self.host, 995)
        self.pop3.user(self.username)
        self.pop3.pass_(self.password)

    def update_latest_id(self):
        self.update()
        _, messages, _ = self.pop3.list()
        if messages:
            latest_email_index = int(messages[-1].split()[0])
            return latest_email_index
        else:
            return None

    @staticmethod
    def parse_massage(msg):
        def guess_charset(msg):
            charset = msg.get_charset()
            if charset is None:
                content_type = msg.get("Content-Type", "").lower()
                pos = content_type.find("charset=")
                if pos >= 0:
                    charset = content_type[pos + 8 :].strip()
            return charset

        def decode_str(s):
            value, charset = decode_header(s)[0]
            if charset:
                value = value.decode(charset)
            return value

        re = {}
        for header in ["From", "To", "Subject"]:
            value = msg.get(header, "")
            if value:
                if header == "Subject":
                    value = decode_str(value)
                else:
                    hdr, addr = parseaddr(value)
                    name = decode_str(hdr)
                    value = (name, addr)
            re[header] = value
        if msg.is_multipart():
            msgs = []
            parts = msg.get_payload()
            for part in parts:
                msgs.append(POP3Server.parse_massage(part))
            re["Content"] = msgs
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain" or content_type == "text/html":
                content = msg.get_payload(decode=True)
                charset = guess_charset(msg)
                if charset:
                    content = content.decode(charset)
            else:
                content = msg.get_payload()
            re["Content"] = (content_type, content)
        return re

    @staticmethod
    def parse_email(lines):
        email_content = b"\n".join(lines).decode("utf-8")
        email_message = Parser().parsestr(email_content)
        return POP3Server.parse_massage(email_message)

    def get_email_by_index(self, index: int):
        """Get the email by index

        Args:
            index (int): The index of the email, -1 means the latest email

        Returns:
            dict: The email content, including "From", "To", "Subject", "Content"
            {"From":(name, addr), "To":(name, addr), "Subject":str, "Content":(content_type, content)|list}
        """
        _, messages, _ = self.pop3.list()
        if messages:
            latest_email_index = int(messages[index].split()[0])
            _, lines, _ = self.pop3.retr(latest_email_index)
            return POP3Server.parse_email(lines)
        else:
            return None

    def get_email_unread(self):
        """Get the latest email

        Returns:
            dict: The email content, including "From", "To", "Subject", "Content"
            {"From":(name, addr), "To":(name, addr), "Subject":str, "Content":(content_type, content)|list}
        """
        lastid = self.update_latest_id()
        if self.lastid is None or self.lastid != lastid:
            self.lastid = lastid
            _, lines, _ = self.pop3.retr(self.lastid)
            return POP3Server.parse_email(lines)
        else:
            return None

    def quit(self):
        self.pop3.quit()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="email title", type=str)
    parser.add_argument("content", help="email content", type=str)
    args = parser.parse_args()
    send_default(args.title, args.content)