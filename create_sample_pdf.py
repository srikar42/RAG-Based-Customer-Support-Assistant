"""
create_sample_pdf.py
--------------------
Generates a sample 'knowledge_base.pdf' so you can test the assistant
without needing a real document.

Run once:
  pip install fpdf2
  python create_sample_pdf.py
"""

try:
    from fpdf import FPDF
except ImportError:
    print("Install fpdf2 first:  pip install fpdf2")
    raise

# ---------------------------------------------------------------------------
# Sample support knowledge-base content
# ---------------------------------------------------------------------------
CONTENT = [
    ("Return & Refund Policy",
     """We accept returns within 30 days of the original purchase date.
Items must be in their original, unused condition with all tags attached.
To initiate a return, log in to your account, navigate to 'My Orders',
select the item, and click 'Request Return'. You will receive a prepaid
shipping label by e-mail within 24 hours.

Refunds are processed within 5-7 business days after we receive the item.
The refund will be credited to the original payment method.
Digital downloads and personalised items are non-refundable."""),

    ("Shipping Information",
     """Standard shipping (5-7 business days): FREE on orders over $50; $4.99 below.
Express shipping (2-3 business days): $12.99.
Overnight shipping (next business day): $24.99.

Orders placed before 2 PM EST on a business day ship the same day.
You will receive a tracking number by e-mail once your order ships.
We currently ship to all 50 US states and Puerto Rico.
International shipping is not available at this time."""),

    ("Account & Password",
     """To reset your password, click 'Forgot Password' on the login page
and enter your registered e-mail address. You will receive a reset link
valid for 60 minutes.

If you no longer have access to your registered e-mail, contact support at
support@example.com with a government-issued ID for identity verification.

To update your e-mail address or personal information, go to
Account Settings -> Personal Information after logging in."""),

    ("Product Warranty",
     """All hardware products come with a 1-year limited warranty covering
manufacturing defects. The warranty does not cover accidental damage,
water damage, or normal wear and tear.

To submit a warranty claim, visit warranty.example.com and provide:
  - Proof of purchase (order number or receipt)
  - Photos of the defect
  - A brief description of the issue

Approved claims will be repaired or replaced at our discretion within
10 business days."""),

    ("Contact Support",
     """Live Chat: Available Monday-Friday, 9 AM-6 PM EST at support.example.com.
E-mail: support@example.com - responses within 1 business day.
Phone: 1-800-555-0199 - Monday-Friday, 9 AM-5 PM EST.

For urgent technical issues outside business hours, use our 24/7 chatbot
at support.example.com/bot."""),
]


def create_pdf(output_path: str = "knowledge_base.pdf") -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Customer Support Knowledge Base", ln=True, align="C")
    pdf.ln(6)

    for heading, body in CONTENT:
        # Section heading
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 9, heading, ln=True, fill=True)
        pdf.ln(2)

        # Body text
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body.strip())
        pdf.ln(6)

    pdf.output(output_path)
    print(f"Sample PDF created: {output_path}")


if __name__ == "__main__":
    create_pdf()
