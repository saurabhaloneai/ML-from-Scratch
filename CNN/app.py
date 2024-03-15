import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image  # Import Image module from PIL
from model import CNN # Import your CNN model class

# Load the trained model
model = CNN(num_classes=5)
model.load_state_dict(torch.load('/Users/sasaurabhurabhvaishubhalone/Desktop/ML-from-scratch/CNN/weights.pth'))
model.eval()


# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define category names
category_names = {0: "Beluga Whale", 1: "Blue Whale", 2: "Fin Whale", 3: "Humpback Whale", 4: "Killer(Orca) Whale"}

def main():
    st.title("CNN Model Deployment")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Get the model predictions
        with torch.no_grad():
            outputs = model(input_batch)
            _, predicted = torch.max(outputs.data, 1)
            confidence_score = torch.softmax(outputs, dim=1).max().item()

        # Check if the image is from the dataset or not
        if confidence_score < 0.5:  # Adjust this threshold as needed
            st.write("This image does not seem to be from the dataset.")
        else:
            # Display the prediction with category name
            predicted_category = category_names[predicted.item()]
            st.write(f"Prediction: {predicted_category}")
            st.write(f"Confidence Score: {confidence_score:.2f}")

if __name__ == "__main__":
    main()