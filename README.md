# Miwel Translate: English-to-Spanish Translator

A complete application that translates English text to Spanish using a custom Transformer neural network model. The application consists of a FastAPI backend that serves the translation model and an Angular frontend for user interaction.

## 📋 Features

- Modern Angular 21 frontend
- FastAPI backend with async support
- Custom Transformer model for English to Spanish translation
- Error handling for invalid inputs
- Responsive design for mobile and desktop

## 🔧 Requirements

### Backend
- Python 3.8+
- PyTorch
- FastAPI
- The pre-trained transformer model (`transformer_final.pth.tar`)

### Frontend
- Node.js 16+
- Angular CLI 16+

## 🚀 Setup and Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/miwel-translate.git
cd miwel-translate
```

### Backend Setup
1. Create a Python virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the model file:
   - Copy `transformer_final.pth.tar` to the `backend/models/` directory
   - Create the directory if it doesn't exist: `mkdir -p models`

### Frontend Setup
1. Install Angular CLI if you don't have it:
```bash
npm install -g @angular/cli
```

2. Install dependencies:
```bash
cd frontend
npm install
```

## 🏃‍♂️ Running the Application

### Run the Backend
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The backend API will be available at http://localhost:8000

### Run the Frontend
```bash
cd frontend
ng serve
```
The frontend application will be available at http://localhost:4200

## 📝 API Documentation

Once the backend is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints
- `GET /health` - Check if the API is running and the model is loaded
- `POST /translate` - Translate English text to Spanish

### Sample Request
```json
{
  "sentence": "hello world"
}
```

### Sample Response
```json
{
  "translation": "hola mundo"
}
```

## 🐳 Docker (Optional)

### Build and run the backend container
```bash
cd backend
docker build -t miwel-translate-backend .
docker run -p 8000:8000 miwel-translate-backend
```

### Build and run the frontend container
```bash
cd frontend
docker build -t miwel-translate-frontend .
docker run -p 4200:80 miwel-translate-frontend
```

## ⚠️ Limitations

- Only lowercase letters and basic punctuation are supported
- Maximum input length is around 300 characters
- The model was trained on a specific dataset and may not handle all translations perfectly

## 📚 Architecture

### Backend
- FastAPI for the REST API
- Pydantic for request/response validation
- PyTorch for running the Transformer model

### Frontend
- Angular 21 with TypeScript
- Reactive Forms for input handling
- HttpClient for API communication
- SCSS for styling
- Proxy configuration to communicate with the backend during development

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.