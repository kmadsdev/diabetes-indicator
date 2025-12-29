# Diabetes Indicator 
**Available at: [https://kmads.dev/diabetes-indicator/](https://kmads.dev/diabetes-indicator/)** <br>
**Backend: [https://source.kmads.dev/diabetes-indicator-backend/](https://source.kmads.dev/diabetes-indicator-backend/)**

Diabetes Indicator is a survey website that asks you 17 questions to determine whether you may have diabetes.

> **⚠️ Disclaimer:** This app is for informational and educational purposes only. It does not provide medical advice, diagnosis, or treatment. The results should not be taken as a medical opinion. If you have concerns about your health or the results, please consult a qualified healthcare professional.

## Features
- **17 Health Questions** - Comprehensive survey covering blood pressure, cholesterol, BMI, lifestyle habits, and more
- **BMI Calculator** - Automatic BMI calculation with metric (kg/cm) and imperial (lbs/ft) support
- **ML-Powered Prediction** - Machine Learning model trained on public health data from Kaggle
- **Mobile Responsive** - Optimized for both desktop and mobile devices
- **Loading Animation** - Visual feedback while analyzing health data
- **Dark Theme** - Modern dark UI design

## Demo
![Demo](https://raw.githubusercontent.com/kmadsdev/diabetes-indicator/main/docs/assets/media/demo-2025-11-17.gif)

## Project Structure
```
diabetes-indicator/
├── index.html      # Main page
├── style.css       # Desktop styles
├── mobile.css      # Mobile responsive styles
├── gateway.js      # API Gateway & form logic
├── assets/
│   ├── banners/    # Header images
│   ├── icons/      # Favicon, warning icons, footer icons
│   ├── media/      # Demo videos/gifs
│   └── infra/      # Infrastructure diagrams
├── LICENSE
└── README.md
```

## Technology Stack
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Python FastAPI (separate repository)
- **ML Model:** Trained on [Diabetes Health Indicators Dataset](https://kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## Changelog

### v1.1.0 (2025-12-29)
- Improved mobile responsiveness (fixed X-axis overflow, centered buttons, consistent spacing)
- New footer with copyright, projects link, and GitHub star button
- Loading animation with animated dots while API processes request
- BMI input placeholders showing metric units (kg/cm or lbs/ft)
- Fixed slider labels alignment for health classification
- Fixed disclaimer layout on mobile devices
- Fixed BMI inputs overflow issues

### v1.0.0 (2025-11-17)
- Initial release
- 17 health survey questions
- BMI calculator with metric/imperial support
- API integration for diabetes prediction

## License
Copyright © 2025 kmads.dev. All rights reserved.
From dev to dev, use it as you'd like, but don't forget to give some credits if possible ❤️
