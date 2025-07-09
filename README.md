# Continual Learning with Learning without Prejudice (LwP)

This project implements a continual learning system from scratch to tackle the problem of catastrophic forgetting. A custom Convolutional Neural Network (CNN) is trained sequentially on a stream of 20 datasets derived from CIFAR-10 without storing past data (except for a small buffer in Task 2).

The core strategy is based on **Learning without Prejudice (LwP)**, which uses knowledge distillation and pseudo-labeling to learn new tasks while preserving previously acquired knowledge.

---

## Project Structure

The project is divided into two main parts, corresponding to the two scripts:

1.  `task1.py`: Implements the continual learning solution for **Task 1**, where there is no domain shift between datasets D1 to D10. It uses the standard LwP strategy.
2.  `task2.py`: Implements the solution for **Task 2**, which involves a domain shift in datasets D11 to D20. This script enhances the LwP strategy with **Experience Replay** to mitigate the negative effects of the changing data distribution.
3.  `data/`: This directory should contain the datasets after downloading and unzipping.
    * `task1_no_domain_shift/`
    * `task2_domain_shift/`

---

## Requirements

* Python 3.x
* PyTorch

No other external libraries like `numpy` are required.

---

## Setup & How to Run

1.  **Download the Dataset:**
    Download the data from the URL provided in the project description:
    `https://tinyurl.com/cs771-mp2-data`

2.  **Unzip the Data:**
    Unzip the file to create a `data` directory with the two sub-folders (`task1_no_domain_shift` and `task2_domain_shift`).

3.  **Place Scripts:**
    Ensure `task1.py` and `task2.py` are in the same root directory as the `data` folder.

4.  **Run the Experiments:**
    Execute the scripts from your terminal.

    * **To run Task 1:**
        ```bash
        python task1.py
        ```

    * **To run Task 2:**
        ```bash
        python task2.py
        ```

---

## Methodology

### Task 1: No Domain Shift

The model is trained sequentially on datasets D1 through D10.
* **Initial Training:** A standard supervised training is performed on the labeled dataset `D1`.
* **Continual Update:** For subsequent tasks (D2-D10), which are unlabeled, the model is updated using the LwP strategy. A copy of the model from the previous step (the "teacher") generates pseudo-labels for the new data. The current model (the "student") is then trained with a combined loss function consisting of:
    1.  **Pseudo-Label Loss:** A standard classification loss on the pseudo-labels.
    2.  **Distillation Loss:** A KL-Divergence loss that encourages the student's output distribution to match the teacher's, preserving past knowledge.

### Task 2: With Domain Shift

The model is trained on datasets D11 through D20, which have changing data distributions. This task builds on the LwP strategy by adding **Experience Replay**.
* **Memory Buffer:** A small buffer stores a few correctly labeled examples from each task after it is learned.
* **Combined Training:** During the update step, the model's total loss is a combination of the LwP losses (from the new, unlabeled data) and a standard supervised loss calculated on a small batch of "replayed" examples from the memory buffer. This replay mechanism acts as an anchor to past knowledge, making the model more robust to domain shifts.

---

## Output

Each script will train the model sequentially and print the progress. At the end of the run, it will display the final **10x10 accuracy matrix**, which shows the model's performance on all previously seen tasks after each update.
