import streamlit as st
from predict import predict
import pandas as pd
import matplotlib.pyplot as plt

# PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import io

st.set_page_config(
    page_title="Cyber Threat Intelligence Analyzer",
    page_icon="🛡",
    layout="wide"
)

st.markdown("""
<style>
.stApp{
background: linear-gradient(120deg,#1d4ed8,#0ea5e9,#06b6d4,#14b8a6);
background-size:400% 400%;
animation: cybermove 12s ease infinite;
color:white;
}

@keyframes cybermove{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.title{
text-align:center;
font-size:90px;
font-weight:bold;
color:#ffffff;
}

.subtitle{
text-align:center;
font-size:40px;
margin-bottom:30px;
color:#f1f5f9;
}

.inputbox{
background:linear-gradient(90deg,#6366f1,#06b6d4);
padding:20px;
border-radius:12px;
font-size:17px;
color:white;
}

.box{
background:rgba(15,23,42,0.75);
padding:20px;
border-radius:12px;
margin-top:15px;
}

.tech{ color:#22c55e; font-size:18px; }
.tactic{ color:#f97316; font-size:18px; }

div.stButton > button, div.stDownloadButton > button {
background: linear-gradient(90deg,#1e3a8a,#2563eb);
color: white;
font-size:16px;
font-weight:bold;
border-radius:10px;
padding:8px 20px;
border:none;
box-shadow:0px 0px 10px rgba(59,130,246,0.7);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Cyber Threat Intelligence Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Detection of MITRE ATTACK Techniques and Tactics</p>', unsafe_allow_html=True)

st.sidebar.title("Project Information")
st.sidebar.info("""
AI-based Cyber Threat Intelligence Analyzer  

Detects **MITRE ATTACK techniques and tactics** from CTI reports using NLP models.
""")

text = st.text_area("Enter Threat Intelligence Report", height=200)
analyze = st.button("Analyze Threat")

if analyze:

    if text.strip() == "":
        st.warning("Please enter a threat report")

    else:
        techniques_with_scores = predict(text)

        techniques = [t[0] for t in techniques_with_scores]
        scores = [t[1] for t in techniques_with_scores]
        tactic_map = {
            "T1595":"Reconnaissance",
            "T1583":"Resource Development",
            "T1189":"Initial Access",
            "T1566":"Initial Access",
            "T1190":"Initial Access",
            "T1059":"Execution",
            "T1047":"Execution",
            "T1204":"Execution",
            "T1547":"Persistence",
            "T1068":"Privilege Escalation",
            "T1027":"Defense Evasion",
            "T1070":"Defense Evasion",
            "T1003":"Credential Access",
            "T1082":"Discovery",
            "T1046":"Discovery",
            "T1021":"Lateral Movement",
            "T1005":"Collection",
            "T1071":"Command and Control",
            "T1105":"Command and Control",
            "T1041":"Exfiltration",
            "T1486":"Impact"
        }

        tactics = []
        for tech in techniques:
            for key in tactic_map:
                if key in tech:
                    tactics.append(tactic_map[key])

        tactics = list(set(tactics))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Threat Intelligence Report")
            st.markdown('<div class="inputbox">', unsafe_allow_html=True)
            st.write(text)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.subheader("Detected Techniques")
            st.markdown('<div class="box">', unsafe_allow_html=True)

            if len(techniques) == 0:
                st.error("No techniques detected")
            else:
                for tech in techniques:
                    st.markdown(f'<p class="tech">⚡ {tech}</p>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Detected Tactics")
        st.markdown('<div class="box">', unsafe_allow_html=True)

        if len(tactics) == 0:
            st.warning("No tactics detected")
        else:
            for tac in tactics:
                st.markdown(f'<p class="tactic">🎯 {tac}</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if len(techniques) > 0:
            st.subheader("Technique Confidence")

            df = pd.DataFrame({
                "Technique": techniques,
                "Confidence": scores
            })

            fig, ax = plt.subplots(figsize=(5, 3))
            colors_list = ["#22c55e","#3b82f6","#facc15","#f97316","#ec4899"]

            ax.bar(df["Technique"], df["Confidence"], color=colors_list[:len(df)])
            ax.set_ylabel("Confidence", fontsize=8)
            ax.set_ylim(0, 1)

            plt.xticks(rotation=30, fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()

            st.pyplot(fig)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            textColor=colors.HexColor("#1d4ed8"),
            alignment=1
        )

        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            textColor=colors.HexColor("#0ea5e9")
        )

        tech_style = ParagraphStyle(
            'TechStyle',
            parent=styles['BodyText'],
            textColor=colors.green
        )

        tactic_style = ParagraphStyle(
            'TacticStyle',
            parent=styles['BodyText'],
            textColor=colors.orange
        )

        content = []

        content.append(Paragraph("🛡 Cyber Threat Intelligence Report", title_style))
        content.append(Spacer(1, 15))

        content.append(Paragraph("📄 Input Threat Report", heading_style))
        table = Table([[Paragraph(text, styles["BodyText"])]], colWidths=[450])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#e0f2fe")),
            ('BOX', (0,0), (-1,-1), 1, colors.blue),
            ('PAD', (0,0), (-1,-1), 10)
        ]))
        content.append(table)
        content.append(Spacer(1, 15))

        content.append(Paragraph("⚡ Detected Techniques", heading_style))
        tech_list = [Paragraph(f"• {t}", tech_style) for t in techniques]
        content.append(ListFlowable(tech_list))
        content.append(Spacer(1, 15))

        content.append(Paragraph("🎯 Detected Tactics", heading_style))
        tac_list = [Paragraph(f"• {t}", tactic_style) for t in tactics]
        content.append(ListFlowable(tac_list))
        content.append(Spacer(1, 15))

        summary = f"""
Detected {len(techniques)} techniques and {len(tactics)} tactics using AI analysis.
These insights help understand attacker behavior and improve defense strategies.
"""
        content.append(Paragraph("📊 Analysis Summary", heading_style))
        content.append(Paragraph(summary, styles["BodyText"]))
        content.append(Spacer(1, 20))

        content.append(Paragraph("Generated by Cyber Threat Intelligence Analyzer 🚀", styles["Italic"]))

        doc.build(content)

        pdf_data = buffer.getvalue()
        buffer.close()

        st.download_button(
            label="📄 Download Advanced PDF Report",
            data=pdf_data,
            file_name="cyber_threat_report.pdf",
            mime="application/pdf"
        )