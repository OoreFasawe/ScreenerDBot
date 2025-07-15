# ScreenerDBot

A Discord bot that posts stock screener results as a table image.

## Deploying to Cloud Run

1. Build and push the container:
   ```
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/screenerdbot
   ```

2. Deploy to Cloud Run:
   ```
   gcloud run deploy screenerdbot \
     --image gcr.io/YOUR_PROJECT_ID/screenerdbot \
     --platform managed \
     --region YOUR_REGION \
     --allow-unauthenticated \
     --set-env-vars DISCORD_TOKEN=your_token_here
   ```