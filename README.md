## 1. Clone the Repository with Submodules

To clone the repository along with its submodules:

```bash
git clone --recurse-submodules https://github.com/newline-lab/agri_neural_mpc.git
cd agri_neural_mpc
```

```bash
git submodule update --init --recursive
```

This ensures all submodules are initialized and updated recursively.

## 2. Set Up Python Environment

It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Python Dependencies

Install the required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Install PyTorch

Choose the appropriate installation based on your system:

- For CUDA 11.8:

  ```bash
  pip install torch==2.0.1+cu118 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- For CPU-only:

  ```bash
  pip install torch torchvision torchaudio
  ```

For other configurations, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 5. Set Up ROS Dependencies

Source your ROS environment:

```bash
source /opt/ros/noetic/setup.bash
```

Then, install ROS package dependencies:

```bash
cd ros
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## 6. Build the Workspace

Build your ROS workspace:

```bash
catkin_make
source devel/setup.bash
```

Or, if you're using `catkin_tools`:

```bash
catkin build
source devel/setup.bash
```


