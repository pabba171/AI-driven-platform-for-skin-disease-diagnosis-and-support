// DOM Elements
const skinImageInput = document.getElementById('skinImage');
const uploadArea = document.getElementById('uploadArea');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeText = document.getElementById('analyzeText');
const analyzeSpinner = document.getElementById('analyzeSpinner');
const resultsContainer = document.getElementById('resultsContainer');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const conditionResult = document.getElementById('conditionResult');
const recommendationResult = document.getElementById('recommendationResult');
const contactForm = document.getElementById('contactForm');
const hamburger = document.getElementById('hamburger');
const navLinks = document.querySelector('.nav-links');
const loadingScreen = document.getElementById('loadingScreen');

// Model and state
let model;
let isModelLoading = true;

// Conditions data
const CONDITIONS = [
  { name: 'Acne', recommendation: 'Use gentle cleansers and avoid oil-based products. Consider seeing a dermatologist for persistent cases.' },
  { name: 'Eczema', recommendation: 'Apply fragrance-free moisturizers regularly. Avoid triggers like harsh soaps and extreme temperatures.' },
  { name: 'Psoriasis', recommendation: 'Use medicated creams with salicylic acid or coal tar. Phototherapy may help in severe cases.' },
  { name: 'Rosacea', recommendation: 'Use gentle skincare products and sunscreen daily. Avoid triggers like spicy foods and alcohol.' },
  { name: 'Healthy Skin', recommendation: 'Maintain your current routine with daily cleansing, moisturizing, and sun protection.' }
];

// Initialize the application
async function init() {
  // Load TensorFlow.js model
  try {
    model = await tf.loadLayersModel('model/model.json');
    console.log('Model loaded successfully');
    isModelLoading = false;
    analyzeBtn.disabled = false;
  } catch (error) {
    console.error('Error loading model:', error);
    alert('Error loading skin analysis model. Please try again later.');
  } finally {
    // Hide loading screen after model loads or fails
    setTimeout(() => {
      loadingScreen.style.opacity = '0';
      setTimeout(() => {
        loadingScreen.style.display = 'none';
      }, 300);
    }, 1000);
  }

  // Setup event listeners
  setupEventListeners();
}

// Set up all event listeners
function setupEventListeners() {
  // Skin image upload
  uploadArea.addEventListener('click', () => skinImageInput.click());
  skinImageInput.addEventListener('change', handleImageUpload);

  // Drag and drop for image upload
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
    uploadArea.style.backgroundColor = 'rgba(42, 127, 140, 0.1)';
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--dark-gray)';
    uploadArea.style.backgroundColor = 'transparent';
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--dark-gray)';
    uploadArea.style.backgroundColor = 'transparent';
    
    if (e.dataTransfer.files.length) {
      skinImageInput.files = e.dataTransfer.files;
      handleImageUpload();
    }
  });

  // Analyze button
  analyzeBtn.addEventListener('click', analyzeSkin);

  // Contact form
  contactForm.addEventListener('submit', handleFormSubmit);

  // Mobile menu toggle
  hamburger.addEventListener('click', toggleMobileMenu);
}

// Handle image upload
function handleImageUpload() {
  if (skinImageInput.files && skinImageInput.files[0]) {
    const file = skinImageInput.files[0];
    
    // Validate file type
    if (!file.type.match('image.*')) {
      alert('Please upload an image file (JPG, PNG)');
      return;
    }
    
    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
      alert('Image size should be less than 5MB');
      return;
    }
    
    // Display file name
    uploadArea.innerHTML = `
      <i class="fas fa-check-circle" style="color: var(--success)"></i>
      <h3>${file.name}</h3>
      <p>Click to change image</p>
    `;
    
    // Enable analyze button if model is loaded
    if (!isModelLoading) {
      analyzeBtn.disabled = false;
    }
  }
}

// Analyze skin image
async function analyzeSkin() {
  if (!skinImageInput.files || !skinImageInput.files[0]) {
    alert('Please upload an image first');
    return;
  }

  // Show loading state
  analyzeText.textContent = 'Analyzing...';
  analyzeSpinner.classList.remove('hidden');
  analyzeBtn.disabled = true;

  try {
    // Load and preprocess image
    const image = await loadImage(skinImageInput.files[0]);
    const tensor = preprocessImage(image);
    
    // Make prediction
    const prediction = model.predict(tensor);
    const results = processPrediction(prediction);
    
    // Display results
    displayResults(results);
    
  } catch (error) {
    console.error('Analysis error:', error);
    alert('Error analyzing image. Please try another image.');
  } finally {
    // Reset button state
    analyzeText.textContent = 'Analyze Now';
    analyzeSpinner.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
}

// Load image as HTMLImageElement
function loadImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

// Preprocess image for model input
function preprocessImage(img) {
  return tf.tidy(() => {
    // Convert to tensor and resize to 224x224 (adjust based on your model)
    let tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat();
    
    // Normalize if your model expects normalized inputs
    // Subtract mean and divide by std (ImageNet stats)
    const mean = [0.485, 0.456, 0.406]; // RGB mean
    const std = [0.229, 0.224, 0.225];  // RGB std
    
    tensor = tensor.div(255.0);
    tensor = tensor.sub(tf.tensor1d(mean).reshape([1, 1, 3]));
    tensor = tensor.div(tf.tensor1d(std).reshape([1, 1, 3]));
    
    return tensor.expandDims(); // Add batch dimension
  });
}

// Process model prediction
function processPrediction(prediction) {
  const scores = prediction.dataSync(); // Get prediction scores as array
  
  // Find the condition with highest probability
  let maxIndex = 0;
  let maxScore = scores[0];
  
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > maxScore) {
      maxScore = scores[i];
      maxIndex = i;
    }
  }
  
  // Return results object
  return {
    condition: CONDITIONS[maxIndex].name,
    confidence: maxScore,
    recommendation: CONDITIONS[maxIndex].recommendation
  };
}

// Display analysis results
function displayResults(results) {
  // Update confidence meter
  const confidencePercent = Math.round(results.confidence * 100);
  confidenceFill.style.width = `${confidencePercent}%`;
  confidenceText.textContent = `${confidencePercent}%`;
  
  // Update confidence meter color based on confidence level
  if (confidencePercent < 50) {
    confidenceFill.style.backgroundColor = 'var(--warning)';
  } else if (confidencePercent < 75) {
    confidenceFill.style.backgroundColor = 'var(--secondary)';
  } else {
    confidenceFill.style.backgroundColor = 'var(--success)';
  }
  
  // Update condition and recommendation
  conditionResult.innerHTML = `<h4>Condition:</h4><p>${results.condition}</p>`;
  recommendationResult.innerHTML = `<h4>Recommendation:</h4><p>${results.recommendation}</p>`;
  
  // Show results container
  resultsContainer.classList.remove('hidden');
  
  // Scroll to results
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Handle contact form submission
function handleFormSubmit(e) {
  e.preventDefault();
  
  const name = document.getElementById('name').value.trim();
  const email = document.getElementById('email').value.trim();
  const message = document.getElementById('message').value.trim();
  
  if (!name || !email || !message) {
    alert('Please fill in all fields');
    return;
  }
  
  // Here you would typically send the data to a server
  // For now, we'll just show a success message
  alert(`Thank you, ${name}! Your message has been sent. We'll contact you soon.`);
  
  // Reset form
  contactForm.reset();
}

// Toggle mobile menu
function toggleMobileMenu() {
  hamburger.classList.toggle('active');
  navLinks.classList.toggle('active');
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
// Updated model loading
const MODELS = {
  CNN: { 
    path: 'models/cnn/model.json',
    description: 'Convolutional Neural Network for image pattern recognition',
    loaded: false,
    instance: null
  },
  RNN: {
    path: 'models/rnn/model.json',
    description: 'Recurrent Neural Network for temporal progression analysis',
    loaded: false,
    instance: null
  },
  GNN: {
    path: 'models/gnn/model.json',
    description: 'Graph Neural Network for lesion relationship mapping',
    loaded: false,
    instance: null
  }
};

// Initialize all models
async function loadAllModels() {
  try {
    const loadingPromises = Object.keys(MODELS).map(async (key) => {
      try {
        MODELS[key].instance = await tf.loadGraphModel(MODELS[key].path);
        MODELS[key].loaded = true;
        console.log(`${key} model loaded successfully`);
      } catch (error) {
        console.error(`Error loading ${key} model:`, error);
        MODELS[key].loaded = false;
      }
    });
    
    await Promise.all(loadingPromises);
    isModelLoading = false;
    analyzeBtn.disabled = false;
  } catch (error) {
    console.error('Error loading models:', error);
  }
}

// Updated analyze function
async function analyzeSkin() {
  const selectedModel = document.querySelector('input[name="modelType"]:checked').value;
  
  if (!MODELS[selectedModel].loaded) {
    alert(`${selectedModel} model is not available. Please try another model.`);
    return;
  }

  // Show loading state
  analyzeText.textContent = `Analyzing with ${selectedModel}...`;
  analyzeSpinner.classList.remove('hidden');
  analyzeBtn.disabled = true;

  try {
    const image = await loadImage(skinImageInput.files[0]);
    let results;
    
    switch(selectedModel) {
      case 'CNN':
        const cnnTensor = preprocessImageForCNN(image);
        results = await analyzeWithCNN(cnnTensor);
        break;
      case 'RNN':
        const rnnTensor = preprocessImageForRNN(image);
        results = await analyzeWithRNN(rnnTensor);
        break;
      case 'GNN':
        const gnnTensor = preprocessImageForGNN(image);
        results = await analyzeWithGNN(gnnTensor);
        break;
    }
    
    displayResults(results);
    compareModels(image); // Optional: Run comparison
    
  } catch (error) {
    console.error(`${selectedModel} analysis error:`, error);
    alert(`Error during ${selectedModel} analysis. Please try again.`);
  } finally {
    analyzeText.textContent = 'Analyze Now';
    analyzeSpinner.classList.add('hidden');
    analyzeBtn.disabled = false;
  }
}

// Model-specific preprocessing
function preprocessImageForCNN(img) {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat();
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    tensor = tensor.div(255.0);
    tensor = tensor.sub(tf.tensor1d(mean).reshape([1, 1, 3]));
    tensor = tensor.div(tf.tensor1d(std).reshape([1, 1, 3]));
    return tensor.expandDims();
  });
}

function preprocessImageForRNN(img) {
  return tf.tidy(() => {
    // RNN might expect a sequence of images, but we'll simulate with patches
    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([256, 256])
      .toFloat();
    
    // Create 8x8 patches as "time steps"
    const patches = tf.image.extractImagePatches(
      tensor.expandDims(),
      [32, 32],
      [32, 32],
      [1, 1],
      'valid'
    );
    
    return patches;
  });
}

function preprocessImageForGNN(img) {
  return tf.tidy(() => {
    // GNN might expect graph-structured data
    // Here we'll create node features from image patches
    const tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([256, 256])
      .toFloat();
    
    // Create node features (16x16 patches)
    const patches = tf.image.extractImagePatches(
      tensor.expandDims(),
      [16, 16],
      [16, 16],
      [1, 1],
      'valid'
    );
    
    // Create adjacency matrix (simplified)
    const numNodes = patches.shape[1] * patches.shape[2];
    const adjacency = tf.ones([numNodes, numNodes]);
    
    return {
      nodeFeatures: patches.reshape([numNodes, -1]),
      adjacency
    };
  });
}

// Model-specific analysis functions
async function analyzeWithCNN(tensor) {
  const prediction = MODELS.CNN.instance.predict(tensor);
  return processPrediction(prediction, 'CNN');
}

async function analyzeWithRNN(tensor) {
  // RNN expects sequence input - using patches as time steps
  const prediction = MODELS.RNN.instance.predict(tensor);
  return processPrediction(prediction, 'RNN');
}

async function analyzeWithGNN(graphData) {
  // GNN expects node features and adjacency matrix
  const prediction = MODELS.GNN.instance.predict([graphData.nodeFeatures, graphData.adjacency]);
  return processPrediction(prediction, 'GNN');
}

// Enhanced prediction processing
function processPrediction(prediction, modelType) {
  const scores = prediction.dataSync();
  const maxIndex = scores.indexOf(Math.max(...scores));
  const confidence = scores[maxIndex];
  
  // Model-specific interpretation
  let condition, recommendation;
  
  if (modelType === 'RNN') {
    condition = `${CONDITIONS[maxIndex].name} Progression`;
    recommendation = `Based on temporal patterns: ${CONDITIONS[maxIndex].recommendation}`;
  } else if (modelType === 'GNN') {
    condition = `${CONDITIONS[maxIndex].name} Network`;
    recommendation = `Based on lesion relationships: ${CONDITIONS[maxIndex].recommendation}`;
  } else {
    condition = CONDITIONS[maxIndex].name;
    recommendation = CONDITIONS[maxIndex].recommendation;
  }
  
  return {
    modelType,
    condition,
    confidence,
    recommendation,
    allScores: scores
  };
}

// Compare all models (optional feature)
async function compareModels(image) {
  if (!Object.values(MODELS).every(m => m.loaded)) return;
  
  const comparisonGrid = document.querySelector('.comparison-grid');
  comparisonGrid.innerHTML = '';
  
  const cnnTensor = preprocessImageForCNN(image);
  const rnnTensor = preprocessImageForRNN(image);
  const gnnTensor = preprocessImageForGNN(image);
  
  const results = await Promise.all([
    analyzeWithCNN(cnnTensor),
    analyzeWithRNN(rnnTensor),
    analyzeWithGNN(gnnTensor)
  ]);
  
  results.forEach(result => {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.innerHTML = `
      <h5>${result.modelType}</h5>
      <p>${result.condition}</p>
      <div class="confidence-bar">
        <div class="confidence-level" style="width: ${Math.round(result.confidence * 100)}%"></div>
      </div>
      <small>${Math.round(result.confidence * 100)}% confidence</small>
    `;
    comparisonGrid.appendChild(card);
  });
  
  document.getElementById('modelComparison').classList.remove('hidden');
}

// Update initialization
async function init() {
  await loadAllModels();
  setupEventListeners();
  
  // Hide loading screen
  setTimeout(() => {
    loadingScreen.style.opacity = '0';
    setTimeout(() => {
      loadingScreen.style.display = 'none';
    }, 300);
  }, 1500);
}
// Contact form data storage
let contactSubmissions = [];

// Existing contact form submission handler - modify it to store data
document.getElementById('contactForm').addEventListener('submit', function(e) {
  e.preventDefault();
  
  const formData = {
    name: document.getElementById('name').value,
    email: document.getElementById('email').value,
    message: document.getElementById('message').value,
    timestamp: new Date().toISOString()
  };
  
  contactSubmissions.push(formData);
  alert('Thank you for your message!');
  this.reset();
});

// Excel export functionality
document.getElementById('downloadContacts').addEventListener('click', function() {
  if (contactSubmissions.length === 0) {
    alert('No contact submissions to export');
    return;
  }

  // Create Excel workbook
  let csvContent = "data:text/csv;charset=utf-8,";
  
  // Add headers
  csvContent += "Name,Email,Message,Date\n";
  
  // Add data rows
  contactSubmissions.forEach(submission => {
    const row = [
      `"${submission.name}"`,
      `"${submission.email}"`,
      `"${submission.message.replace(/"/g, '""')}"`,
      `"${submission.timestamp}"`
    ].join(',');
    csvContent += row + "\n";
  });
  
  // Create download link
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', 'skin_in_contact_submissions.csv');
  document.body.appendChild(link);
  
  // Trigger download
  link.click();
  document.body.removeChild(link);
});
document.getElementById('downloadContacts').addEventListener('click', async function() {
  const password = prompt('Enter admin password:');
  if (!password) return;
  
  try {
    const response = await fetch('contact-export.php', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ password })
    });
    
    if (!response.ok) throw new Error('Export failed');
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'skin_in_contacts.xlsx';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } catch (error) {
    alert('Error exporting contacts: ' + error.message);
  }
});