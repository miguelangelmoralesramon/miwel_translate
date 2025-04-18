#!/bin/bash

echo "Setting up Miwel Translate project..."

# Setup backend
echo "Setting up backend..."
cd backend
chmod +x setup_env.sh
./setup_env.sh
cd ..

# Setup frontend
echo "Setting up frontend..."
cd frontend
chmod +x setup.sh
./setup.sh
cd ..

echo "Project setup complete!"
echo ""
echo "To start the application:"
echo "1. Start the backend:"
echo "   cd backend"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "   uvicorn main:app --reload"
echo ""
echo "2. In a separate terminal, start the frontend:"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "The application will be available at http://localhost:4200"