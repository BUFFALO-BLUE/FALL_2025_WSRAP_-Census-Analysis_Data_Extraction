import os

def hpc_setup_checklist():
    """Checklist for UConn HPC setup"""
    
    print("=== UCONN HPC SETUP CHECKLIST ===")
    
    checklist = [
        "✅ Create HPC account (if not already done)",
        "🔲 Transfer project code to HPC",
        "🔲 Set up Python environment on HPC", 
        "🔲 Install required libraries (OpenCV, TensorFlow/PyTorch)",
        "🔲 Transfer processed dataset to HPC",
        "🔲 Create SLURM job scripts for training",
        "🔲 Test small training run",
        "🔲 Scale up to full dataset"
    ]
    
    print("HPC Preparation Steps:")
    for item in checklist:
        print(f"  {item}")
    
    print("\n📞 UConn HPC Resources:")
    print("  - Website: https://hpc.uconn.edu/")
    print("  - Support: hpc@uconn.edu")
    print("  - Documentation: https://hpc.uconn.edu/documentation/")
    
    print("\n🚀 Recommended HPC Strategy:")
    print("  1. Start with CPU nodes for data preprocessing")
    print("  2. Use GPU nodes for model training")
    print("  3. Request adequate storage for dataset")
    print("  4. Use SLURM for job scheduling")

def create_hpc_requirements():
    """Create requirements file for HPC"""
    
    requirements = """
# HPC Environment Requirements
opencv-python==4.8.1.78
tensorflow==2.13.0
# or pytorch if preferred
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
Pillow==10.0.0
"""
    
    with open('hpc_requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created hpc_requirements.txt")
    print("Use this to set up your HPC environment")

if __name__ == "__main__":
    hpc_setup_checklist()
    create_hpc_requirements()