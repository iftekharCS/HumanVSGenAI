import pandas as pd
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Creates one continuous list for human_code and ai_code:
        - human_code = [Sol-1 entries..., then Sol-2 entries..., then Sol-3 entries...]
        - ai_code = [AI Sol-1 entries..., AI Sol-2..., AI Sol-3...]
        """
        self.data = pd.read_csv(csv_file)
        
        # Create one continuous list for human solutions
        self.human_code = []
        self.human_code.extend(self.data['Sol-1'].dropna().astype(str).tolist())
        self.human_code.extend(self.data['Sol-2'].dropna().astype(str).tolist())
        self.human_code.extend(self.data['Sol-3'].dropna().astype(str).tolist())
        
        # Create one continuous list for AI solutions
        self.ai_code = []
        self.ai_code.extend(self.data['AI Sol-1'].dropna().astype(str).tolist())
        self.ai_code.extend(self.data['AI Sol-2'].dropna().astype(str).tolist())
        self.ai_code.extend(self.data['AI Sol-3'].dropna().astype(str).tolist())
    
    def __len__(self):
        return len(self.human_code)  # All lists will be same length
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        
        return {
            'human_code': self.human_code[idx],
            'ai_code': self.ai_code[idx]
        }

# Example usage
if __name__ == "__main__":
    dataset = CodeDataset('dataset1.csv')
    
    print(f"Total entries: {len(dataset)}")