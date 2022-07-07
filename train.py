import os, json, logging, argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import DatasetForLM
from model import ReformerLM


logger = logging.getLogger(__name__)

def main(args):
    # Prepare dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = DatasetForLM(tokenizer, args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch)
    
    args.vocab = len(tokenizer)

    model = ReformerLM(args)
    model.to(args.device)


    # total iteration and batch size
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.train_epochs
    total_batch_size = args.train_batch * args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    tr_loss = 0.0
    global_steps = 0
    local_steps = 0
    model.zero_grad()

    print("Start Training! ------")
    model.train()
    for epoch in range(int(args.train_epochs)):
        for step, batch in tqdm(enumerate(train_dataloader)):
            inputs, labels, inputs_mask = batch
            inputs, labels, inputs_mask = inputs.to(args.device), labels.to(args.device), inputs_mask.to(args.device)

            _, loss = model(inputs, labels=labels)
            loss.backward()

            tr_loss += loss.item()
            global_steps += 1
            local_steps += 1

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

            if (step+1) % args.logging_steps == 0:
                log_text= f"Train Loss: {tr_loss/local_steps} | Epochs: {epoch+1} | Steps: {global_steps}"
                with open(f'training_log-bl{args.bucket_length}_nl{args.num_layers}_dk{args.d_model}_sl{args.max_len}.txt', 'a+') as log_file:
                    log_file.write(log_text+"\n")
                    log_file.close()
                tr_loss = 0.0
                local_steps = 0



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # model config setup
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased')
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--bucket_length", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--causal", type=bool, default=True)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--chunks", type=int, default=8)

    # training setup 
    parser.add_argument("--train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # logging/save setup
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")

    args = parser.parse_args()

    device = torch.device("cuda")
    args.n_gpu = 1
    args.device = device

    main(args)
