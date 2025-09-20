# Grid Navigation
> AI learning to navigate the grid world

## Getting Started
1. **Clone the repository**
```bash
git clone https://github.com/aditya-shriwastava/grid_nav.git
cd grid_nav
```

2. **Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -e .
```

4. **Usage**
After installation, you can run grid nav with:
```bash
grid-nav
```

## Behaviour Cloning Results

| Number of Trajectories (Train) | Success Percentage (Train) | Success Percentage (Test) |
|:------------------------------:|:--------------------------:|:-------------------------:|
| 0.1k                           | 10.1%                      | 7.0%                      |
| 1k                             | 87.3%                      | 42.9%                     |
| 10k                            | 99.8%                      | 40.5%                     |
| 50k                            | 99.6%                      | 43.9%                     |
| 100k                           | 99.3%                      | 42.2%                     |                

**Major issue: Agent oscillate between two states and get in loop it can never escape!**
