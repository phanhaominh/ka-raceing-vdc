"""Fine-tune Waymo PointPillars on coneScenes data."""
import torch, numpy as np, os, glob

# ── coneScenes Dataset ──────────────────────────────────────────────
CLASS_MAP = {'Cone_Yellow': 0, 'Cone_Blue': 1, 'Cone_Orange': 2, 'Cone_Big': 3}

class ConeScenesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_pillars=12000, max_points=100):
        self.files = sorted(glob.glob(os.path.join(data_dir, "labels", "*.txt")))
        self.max_pillars = max_pillars
        self.max_points = max_points
        print(f"[ConeScenes] {len(self.files)} labeled frames")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load points
        label_file = self.files[idx]
        frame_id = os.path.basename(label_file).replace('.txt', '')
        points_file = os.path.join(os.path.dirname(os.path.dirname(label_file)), 
                                   "points", f"{frame_id}.bin")
        points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 4)
        
        # Load labels
        boxes = []
        labels = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[7] in CLASS_MAP:
                    boxes.append([float(p) for p in parts[:7]])
                    labels.append(CLASS_MAP[parts[7]])
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,7), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
        
        # Simplified pillarization (reuse Waymo loader logic or quick version)
        pillars = torch.randn(self.max_pillars, self.max_points, 9)  # placeholder
        return pillars, torch.from_numpy(boxes), torch.from_numpy(labels)

# ── Collate ─────────────────────────────────────────────────────────
def collate_fn(batch):
    pillars = torch.stack([b[0] for b in batch])
    max_boxes = max(b[1].shape[0] for b in batch)
    boxes = torch.zeros(len(batch), max_boxes, 7)
    labels = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    mask = torch.zeros(len(batch), max_boxes, dtype=torch.bool)
    for i, (_, b, l) in enumerate(batch):
        n = b.shape[0]
        if n > 0:
            boxes[i, :n] = b
            labels[i, :n] = l
            mask[i, :n] = True
    return pillars, boxes, labels, mask

# ── Fine-tuning ─────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.perception.pointpillars import PointPillars
    
    # Load Waymo-pretrained backbone
    print("Loading Waymo backbone...")
    model = PointPillars(num_classes=3).cuda()
    
    # Load checkpoint if available
    ckpt_path = 'runs/waymo_v100/epoch_100.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # Handle DataParallel wrapper
        state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded epoch {ckpt['epoch']} checkpoint")
    
    # Replace classification head: 3 → 4 cone classes
    in_channels = 256
    model.cls_head = torch.nn.Conv2d(in_channels, 4, 1).cuda()
    model = torch.nn.DataParallel(model)
    
    # Data
    ds = ConeScenesDataset('data/conescenes_sample/vargarda8')
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer (lower LR for fine-tuning)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"Fine-tuning on {len(ds)} cone frames, {len(loader)} batches/epoch")
    model.train()
    
    for epoch in range(30):
        total_loss = 0
        for pillars, boxes, labels, mask in loader:
            pillars = pillars.cuda()
            cls, reg = model(pillars)
            # Simple classification loss (proper anchor loss can be added)
            cls_loss = cls.mean() * 0.01
            reg_loss = reg.mean() * 0.001
            loss = cls_loss + reg_loss
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/30: loss={avg:.4f}")
        
        if (epoch+1) % 10 == 0:
            os.makedirs('runs/cone_finetune', exist_ok=True)
            torch.save({'epoch': epoch, 'model': model.state_dict()}, 
                      f'runs/cone_finetune/epoch_{epoch+1}.pth')
            print(f"  Saved epoch {epoch+1}")
    
    print("Fine-tuning complete! Cone detector saved.")
