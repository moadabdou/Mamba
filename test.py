import torch
from torch import nn, optim
from torch.nn import functional as F
from model import BasicMambaModel
import tiktoken
import time
import sys
import glob

# Conversation dataset
text_data = ""
with open("./conversation.txt", "r") as f:
    text_data = f.read()

# Hyperparameters
emb_dim = 128
up_proj_dim = 256
state_space_dim = 16
n_blocks = 4
learning_rate = 1e-3
training_steps = 100
batch_size = 16
context_length = 256

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab


def create_model():
    """Initialize the Mamba model with global hyperparameters."""
    return BasicMambaModel(
        n_blocks=n_blocks,
        up_proj_dim=up_proj_dim,
        state_space_dim=state_space_dim,
        emb_dim=emb_dim,
        vocab_size=vocab_size
    ).to(device)


def train_model():
    """Full training process."""
    print("\n--- 🚀 Starting New Training Session ---")

    full_data = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)
    print(f"✅ Training data: {len(full_data)} tokens")

    def get_batch():
        ix = torch.randint(0, len(full_data) - context_length, (batch_size,))
        x = torch.stack([full_data[i:i + context_length] for i in ix])
        y = torch.stack([full_data[i + 1:i + context_length + 1] for i in ix])
        return x.to(device), y.to(device)

    model = create_model()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model initialized with {num_params:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for step in range(training_steps):
        xb, yb = get_batch()
        logits = model(xb)
        B, L, C = logits.shape
        loss = loss_fn(logits.view(B * L, C), yb.view(B * L))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 4 == 0 or step == training_steps - 1:
            print(f"Step {step:4d}/{training_steps}, Loss: {loss.item():.4f}")

    end_time = time.time()
    print(f"--- ✅ Training finished in {end_time - start_time:.2f} seconds ---")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = f"mamba_conv_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to '{model_save_path}'")

    return model


@torch.no_grad()
def generate_stream(model, full_context_text, max_new_tokens):
    """Stream tokens autoregressively using full conversation history."""
    model.eval()
    x = torch.tensor(tokenizer.encode(full_context_text), dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= context_length else x[:, -context_length:]
        logits = model(x_cond)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, idx_next), dim=1)
        yield tokenizer.decode([idx_next.item()])


def run_chat(model):
    """Interactive chat loop with conversation history."""
    print("\n--- 💬 Starting Interactive Chat ---")
    print("Enter your prompt. Type 'quit' or 'exit' to end the session.")

    history = ""
    while True:
        try:
            user_prompt = input("\nUser: ")
            if user_prompt.lower() in ["quit", "exit"]:
                print("Assistance: Goodbye! 👋")
                break

            history += f"User: {user_prompt}\nAssistance:"
            print("Assistance: ", end="")

            response_text = ""
            for token in generate_stream(model, history, max_new_tokens=60):
                print(token, end="")
                sys.stdout.flush()
                response_text += token
                if "\n" in token:
                    break

            history += response_text
        except KeyboardInterrupt:
            print("\nAssistance: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break


if __name__ == "__main__":
    print(f"✅ Using device: {device}")
    print(f"✅ Tokenizer: GPT-2 (vocab size {vocab_size})")

    saved_models = glob.glob("*.pth")
    model = None

    if saved_models:
        print("\nFound saved models:")
        for i, model_path in enumerate(saved_models):
            print(f"  {i}: {model_path}")

        while True:
            choice = input(f"Choose a model (0-{len(saved_models)-1}) or 'T' to train new: ").strip().upper()
            if choice == 'T':
                model = train_model()
                break
            try:
                idx = int(choice)
                if 0 <= idx < len(saved_models):
                    path = saved_models[idx]
                    print(f"--- 💾 Loading {path} ---")
                    model = create_model()
                    model.load_state_dict(torch.load(path, map_location=device))
                    print("✅ Model loaded successfully")
                    break
                else:
                    print("Invalid number. Try again.")
            except ValueError:
                print("Invalid input. Enter a number or 'T'.")
    else:
        print("\nNo saved models found.")
        if input("Train a new model? (y/n): ").strip().lower() == 'y':
            model = train_model()

    if model:
        run_chat(model)
