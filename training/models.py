import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import torch
from meld_dataset import MELDDataset
from sklearn.metrics import accuracy_score, precision_score
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        for param in self.bert.parameters():
            param.requires_grad = False #Makes the params not trainable. 
        
        self.projection = nn.Linear(768, 128)
        
    def forward(self, input_ids, attention_mask):
        #Extract Bert Embeddings
        outputs = self.bert(input_ids, attention_mask) 
        #Turns the token ids into embeddings 
        
        #use [CLS] token 
        pooled_output = outputs.pooler_output
        
        return self.projection(pooled_output)
        
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, video_frames):
        video_frames = video_frames.transpose(1,2) 
        return self.backbone(video_frames)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        for param in self.conv_layers.parameters():
            param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(64, 128 ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, audio_features):
        audio_frames = audio_features.squeeze(1)
        features = self.conv_layers(audio_frames)
        
        return self.projection(features.squeeze(-1))
    
class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        #Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        #Classificaion Heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, text_inputs, video_input, audio_input):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        
        video_features = self.video_encoder(video_input)
        audio_features = self.audio_encoder(audio_input)
        
        #Concatenate features
        
        combined_features = torch.cat([text_features, video_features, audio_features], dim=1)
        
        #Fusion Layer
        fused_features = self.fusion_layer(combined_features)
        
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)
        
        return {
            "emotion_output": emotion_output,
            "sentiment_output": sentiment_output
        }
    
    
    
class MultiModalTrainer: 
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        #Dataset Sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        
        print("Dataset Sizes")
        print(f"Train Size: {train_size}")
        print(f"Validation Size: {val_size}")
        print(f"Batches per Epoch: {len(train_loader):,}")
        
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        #If running on AWS SageMaker, use the model directory for tensorboard logs
        base_dir = "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        log_dir = f"{base_dir}/run/{timestamp}"
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        #A way to track the epoch you have, so you can associate specific logs with the epoch
        
        
        
        #Loss Function
        self.optimizer = torch.optim.Adam([
            {"params": self.text_encoder.parameters(), "lr": 8e-6},
            {"params": self.video_encoder.parameters(), "lr": 8e-5},
            {"params": self.audio_encoder.parameters(), "lr": 8e-5},
            {"params": self.fusion_layer.parameters(), "lr": 5e-4},
            {"params": self.emotion_classifier.parameters(), "lr": 5e-5},
            {"params": self.sentiment_classifier.parameters(), "lr": 8e-6},
        ], weight_decay = 1e^-5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimzer,
            model="min",
            factor = 0.1,
            patience = 2
        )
        
        self.current_train_losses = None
        
        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing = 0.05
        )
        
        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing = 0.05
        )
        
    def log_metrics(self, losses, metrics, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else:
            self.writer.add_scalar(
                "loss/total/train", self.current_train_losses['total'], self.global_step
            )
            self.writer.add_scalar(
                "loss/total/val", self.current_val_losses["total"], self.global_step
            )
            
            self.writer.add_scalar(
                "loss/emotion/train", self.current_train_losses["emotion"], self.global_step
            )
            self.writer.add_scalar(
                "loss/emotion/val", self.current_val_losses["emotion"], self.global_step
            )
            self.writer.add_scalar(
                "loss/sentiment/train", self.current_train_losses["sentiment"], self.global_step
            )
            self.writer.add_scalar(
                "loss/sentiment/val", self.current_val_losses["sentiment"], self.global_step
            )
            
        if metrics:
            self.writer.add_scalar(
                f"{phase}/emotion_precision", metrics["emotion_precision"], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/emotion_accuracy", metrics["emotion_accuracy"], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/sentiment_precision", metrics["sentiment_precision"], self.global_step
            )
            self.writer.add_scalar(
                f"{phase}/sentiment_accuracy", metrics["sentiment_accuracy"], self.global_step
            )
        

        
        
        
    
    def train_epoch(self):
        self.model.train()    
        running_loss = {"total": 0, "emotion": 0, "sentiment": 0}
        
        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
            }
            
            video_frames = batch["video_frames"].to(device)
            audio_features = batch["audio_features"].to(device)
            
            emotion_labels = batch["emotion_labels"].to(device)
            sentiment_labels = batch["sentiment_labels"].to(device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(text_inputs, video_frames, audio_features)
            
            emotion_loss = self.emotion_criterion(outputs["emotion_output"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs["sentiment_output"], sentiment_labels)
            
            total_loss = emotion_loss + sentiment_loss
            
            #Batckward Pass
            total_loss.backward()
            #Gradient Clipping: Scale down the gradient to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            #Track Losses
            running_loss["total"] += total_loss.item()
            running_loss["emotion"] += emotion_loss.item()
            running_loss["sentiment"] += sentiment_loss.item()
            
            self.log_metrics(
                "total": total_loss.item(),
                "emotion": emotion_loss.item(),
                "sentiment": sentiment_loss.item()
            )
            
            self.global_step +=1 
        
        return {k: v/len(self.train_loader) for k,v in running_loss.items()} #Average Losses Per Batch
    
    def validate_epoch(self, data_loader, phase='val'):
        self.model.eval() #Disables Dropout       
        val_loss = {"total": 0, "emotion": 0, "sentiment": 0}
        
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []
        
        with torch.inference_mode():
            for batch in self.data_loader:
                device = next(self.model.parameters()).device #Figues out whih device (cpu or gpu) the model is currently on
                text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device)
                }
                
                video_frames = batch["video_frames"].to(device)
                audio_features = batch["audio_features"].to(device)
                
                emotion_labels = batch["emotion_labels"].to(device)
                sentiment_labels = batch["sentiment_labels"].to(device)
                
                outputs = self.model(text_inputs, video_frames, audio_features)
                
                emotion_loss = self.emotion_criterion(outputs["emotion_output"], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs["sentiment_output"], sentiment_labels)        
                total_loss = emotion_loss + sentiment_loss
                    
                all_emotion_preds.append(outputs["emotion_output"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(
                    emotion_labels.cpu().numpy()
                )
                
                all_sentiment_preds.append(outputs["sentiment_output"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(
                    sentiment_labels.cpu().numpy()
                )
                
                val_loss["total"] += total_loss.item()
                val_loss["emotion"] += emotion_loss.item()
                val_loss["sentiment"] += sentiment_loss.item()
            
        avg_loss = {
            k: v/len(self.data_loader) for k,v in val_loss.items()
        }
        
        emotion_precision = precision_score(
            all_emotion_labels,
            all_emotion_preds,
            average="weighted"
        )
        emotion_accuracy = accuracy_score(
            all_emotion_labels,
            all_emotion_preds
        )
        
        sentiment_precision = precision_score(
            all_sentiment_labels,
            all_sentiment_preds,
            average="weighted"
        )
        
        sentiment_accuracy = accuracy_score(
            all_sentiment_labels,
            all_sentiment_preds
        )
        
        self.log_metrics(
            {"emotion_preicison": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy},
            phase=phase
        )
        
        if phase == "val":
            self.scheduler.step(avg_loss["total"])
        
        return {
            "emotion_precision": emotion_precision,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_precision": sentiment_precision,
            "sentiment_accuracy": sentiment_accuracy
        }
            
            

if __name__ == "__main__":
    dataset = MELDDataset(
        csv_path="../dataset/train/train_sent_emo.csv",
        video_directory="../dataset/train/train_splits"
    )
    
    sample = dataset[0]
    model = MultiModalSentimentModel()
    model.eval()
    
    text_inputs = {
        "input_ids": sample["text_inputs"]["input_ids"].unsqueeze(0),
        "attention_mask": sample["text_inputs"]["attention_mask"].unsqueeze(0)
    }
    
    video_frames = sample["video_frames"].unsqueeze(0)
    audio_features = sample["audio_features"].unsqueeze(0)
    
    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        
        emotion_probs = torch.softmax(outputs["emotion_output"], dim=1)[0]
        sentiment_probs = torch.softmax(outputs["sentiment_output"], dim=1)[0]
        
        emotion_map = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise"
            }
        
        sentiment_map = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        
        for i, prob in enumerate(emotion_probs):
            print(f"Emotion: {emotion_map[i]}, Probability: {prob.item():.2f}")
        
        for i, prob in enumerate(sentiment_probs):
            print(f"Sentiment: {sentiment_map[i]}, Probability: {prob.item():.2f}")
    
    print('Predictions for Utterance')