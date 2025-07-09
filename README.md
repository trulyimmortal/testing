# QuestionCraft AI Backend

AI-powered question generation backend using HuggingFace Transformers with the `valhalla/t5-base-qg-hl` model.

## Features

- **Universal Subject Support**: Works with any content without predefined subjects
- **Advanced AI Model**: Uses T5-based question generation model
- **Fast Processing**: Optimized for quick question generation
- **RESTful API**: Simple HTTP endpoints for integration
- **Free Deployment**: Designed for Railway/Render free tiers

## API Endpoints

### Health Check
\`\`\`
GET /health
\`\`\`

### Generate Questions
\`\`\`
POST /generate-questions
Content-Type: application/json

{
  "content": "Your text content here",
  "max_questions": 10
}
\`\`\`

Response:
\`\`\`json
{
  "questions": ["Generated question 1?", "Generated question 2?"],
  "total_generated": 2,
  "processing_time": 1.23,
  "model_info": {
    "model_name": "valhalla/t5-base-qg-hl",
    "device": "cpu",
    "framework": "transformers"
  }
}
\`\`\`

## Local Development

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the server:
\`\`\`bash
uvicorn main:app --reload
\`\`\`

3. Test the API:
\`\`\`bash
python test_api.py
\`\`\`

## Deployment

### Railway Deployment

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.toml` configuration
3. Deploy with one click

### Render Deployment

1. Connect your GitHub repository to Render
2. Use the `render.yaml` configuration
3. Deploy as a web service

## Model Information

- **Model**: `valhalla/t5-base-qg-hl`
- **Type**: T5-based question generation
- **Size**: ~220MB
- **Languages**: English
- **Specialization**: Highlight-based question generation

## Performance

- **Cold Start**: ~30-60 seconds (model loading)
- **Warm Requests**: ~1-3 seconds per request
- **Memory Usage**: ~1-2GB RAM
- **Concurrent Requests**: Supports multiple simultaneous requests

## Supported Content Types

- Marketing materials
- Legal documents
- Technical manuals
- Academic papers
- Business reports
- Any English text content

## Error Handling

The API includes comprehensive error handling:
- Input validation
- Model loading errors
- Generation failures
- Graceful fallbacks

## Monitoring

- Health check endpoint for uptime monitoring
- Detailed logging for debugging
- Performance metrics in responses
