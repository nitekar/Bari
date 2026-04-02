/**
 * pdfService.ts — Generate and share PDF referral letters
 *
 * Uses expo-print for HTML → PDF conversion.
 * Uses expo-sharing on native (share sheet) and browser download on web.
 *
 * SECURITY: All user-supplied data is HTML-escaped before interpolation
 * to prevent template injection / XSS in the PDF WebView.
 */
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';
import { colors } from '../shared/theme/colors';

export interface ReferralData {
  patientAge: number;
  patientGender: 'Male' | 'Female';
  screeningDate: string; // ISO 8601
  prediction: string;
  confidence: number;
  hbLevel?: number | null;
  referralAction: string;
}

/**
 * Escape HTML special characters to prevent injection.
 */
function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Generate HTML content for the referral letter.
 * All dynamic values are HTML-escaped for security.
 */
function generateReferralHTML(data: ReferralData): string {
  const severityColor =
    data.prediction === 'Severe'
      ? '#E57373'
      : data.prediction === 'Moderate'
      ? '#FFB74D'
      : data.prediction === 'Mild'
      ? '#FFD54F'
      : '#81C784';

  const formattedDate = new Date(data.screeningDate).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  // Escape all user-supplied values
  const safePrediction = escapeHtml(data.prediction);
  const safeAge = escapeHtml(data.patientAge);
  const safeGender = escapeHtml(data.patientGender);
  const safeConfidence = escapeHtml(Math.round(data.confidence * 100));
  const safeDate = escapeHtml(formattedDate);
  const safeReferralAction = escapeHtml(data.referralAction);
  const safeHbLevel = data.hbLevel != null ? escapeHtml(data.hbLevel) : null;

  return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          padding: 40px;
          color: #333;
          line-height: 1.6;
        }
        .header {
          text-align: center;
          border-bottom: 3px solid ${colors.primary};
          padding-bottom: 20px;
          margin-bottom: 30px;
        }
        .header h1 {
          margin: 0;
          color: ${colors.primaryDark};
          font-size: 24px;
        }
        .header p {
          color: #666;
          margin: 5px 0 0;
        }
        .severity-badge {
          display: inline-block;
          background-color: ${severityColor};
          color: white;
          padding: 8px 20px;
          border-radius: 20px;
          font-weight: bold;
          font-size: 18px;
          margin: 10px 0;
        }
        .section {
          margin-bottom: 20px;
        }
        .section h2 {
          color: ${colors.primaryDark};
          font-size: 16px;
          border-bottom: 1px solid #eee;
          padding-bottom: 5px;
        }
        .field {
          display: flex;
          margin-bottom: 8px;
        }
        .field-label {
          font-weight: 600;
          min-width: 180px;
          color: #555;
        }
        .field-value {
          color: #333;
        }
        .urgency-box {
          background-color: ${severityColor}22;
          border-left: 4px solid ${severityColor};
          padding: 15px;
          margin: 15px 0;
          border-radius: 4px;
        }
        .footer {
          margin-top: 40px;
          padding-top: 20px;
          border-top: 1px solid #eee;
          text-align: center;
          color: #999;
          font-size: 12px;
        }
        .emergency {
          background-color: #FFEBEE;
          border: 1px solid #E57373;
          padding: 15px;
          border-radius: 8px;
          text-align: center;
          margin-top: 20px;
        }
        .emergency strong {
          color: #C62828;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>Anemia Screening — Referral Letter</h1>
        <p>AI-Powered Conjunctiva Analysis Report</p>
      </div>

      <div style="text-align: center;">
        <span class="severity-badge">${safePrediction}</span>
        <p>Confidence: ${safeConfidence}%</p>
      </div>

      <div class="section">
        <h2>Patient Information</h2>
        <div class="field">
          <span class="field-label">Age (months):</span>
          <span class="field-value">${safeAge}</span>
        </div>
        <div class="field">
          <span class="field-label">Gender:</span>
          <span class="field-value">${safeGender}</span>
        </div>
        ${safeHbLevel != null ? `
        <div class="field">
          <span class="field-label">Hemoglobin Level:</span>
          <span class="field-value">${safeHbLevel} g/dL</span>
        </div>
        ` : ''}
        <div class="field">
          <span class="field-label">Screening Date:</span>
          <span class="field-value">${safeDate}</span>
        </div>
      </div>

      <div class="section">
        <h2>Screening Result</h2>
        <div class="urgency-box">
          <p><strong>Classification:</strong> ${safePrediction}</p>
          <p><strong>Recommendation:</strong> ${safeReferralAction}</p>
        </div>
      </div>

      ${data.prediction === 'Severe' || data.prediction === 'Moderate' ? `
      <div class="emergency">
        <strong>Rwanda Health Emergency Hotline</strong><br />
        <strong style="font-size: 20px;">+250 800 22 333</strong><br />
        <span style="color: #666; font-size: 12px;">Available 24/7 for health emergencies</span>
      </div>
      ` : ''}

      <div class="footer">
        <p>Generated by Bari Anemia Screening App v2.0.0</p>
        <p>This report is AI-generated and should be reviewed by a qualified healthcare professional.</p>
        <p>Report generated on: ${safeDate}</p>
      </div>
    </body>
    </html>
  `;
}

/**
 * Generate a PDF referral letter and share/download it.
 */
export async function generateAndShareReferralPDF(data: ReferralData): Promise<void> {
  const html = generateReferralHTML(data);

  // Generate PDF
  const { uri } = await Print.printToFileAsync({
    html,
    base64: false,
  });

  // Share based on platform
  if (Platform.OS === 'web') {
    // On web, open the PDF in a new tab for download
    window.open(uri, '_blank');
  } else {
    // On native, use the share sheet
    const isShareAvailable = await Sharing.isAvailableAsync();
    if (isShareAvailable) {
      await Sharing.shareAsync(uri, {
        mimeType: 'application/pdf',
        dialogTitle: 'Share Referral Letter',
        UTI: 'com.adobe.pdf',
      });
    }
  }
}
