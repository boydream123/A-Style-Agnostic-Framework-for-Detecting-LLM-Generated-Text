import requests
import json
import pandas as pd
import time
import random
from datasets import load_dataset
from tqdm import tqdm
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalDatasetBuilder:
    def __init__(self, config_file="api_config.json"):
        """
        Initialize universal dataset builder
        
        Args:
            config_file: Path to API configuration file
        """
        self.config = self.load_config(config_file)
        self.dataset = []
        
        # Default rewriting prompt (can be customized in config)
        self.default_prompt = """Rewrite this text in a different style, transforming it into Hans Christian Andersen's fairy tale writing style.
        When rewriting, please appropriately introduce the following characteristics:
        1. Uniform language style
        2. Overly structured and logical framework
        3. Lack of background knowledge and personal experience
        4. Repetitive or formulaic expressions
        5. Potential data bias and errors
        
        Original text: {text}
        
        Please output the rewritten text directly:"""
    
    def load_config(self, config_file):
        """
        Load API configuration from JSON file
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "api_url": "http://111.172.228.182:5200/api/v1/chat/completions",
            "model_name": "deepseek-ai/DeepSeek-V3",
            "api_key": "",
            "headers": {
                "Content-Type": "application/json"
            },
            "request_format": "openai",  # openai, anthropic, custom
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1,
            "rate_limit_delay": 1,
            "custom_prompt": ""
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        else:
            # Create default config file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Created default configuration file: {config_file}")
        
        return default_config
    
    def get_rewrite_prompt(self):
        """Get the rewriting prompt from config or use default"""
        return self.config.get("custom_prompt", "") or self.default_prompt
    
    def prepare_api_request(self, text):
        """
        Prepare API request based on configuration
        
        Args:
            text: Text to rewrite
            
        Returns:
            headers and data for API request
        """
        headers = self.config["headers"].copy()
        
        # Add API key to headers if provided
        if self.config.get("api_key"):
            if self.config["request_format"] == "openai":
                headers["Authorization"] = f"Bearer {self.config['api_key']}"
            elif self.config["request_format"] == "anthropic":
                headers["x-api-key"] = self.config["api_key"]
            # Add more API formats as needed
        
        prompt = self.get_rewrite_prompt().format(text=text)
        
        # Prepare request data based on format
        if self.config["request_format"] == "openai":
            data = {
                "model": self.config["model_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"]
            }
        elif self.config["request_format"] == "anthropic":
            data = {
                "model": self.config["model_name"],
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        elif self.config["request_format"] == "custom":
            # For custom API formats, use the template from config
            data = self.config.get("request_template", {})
            # Replace placeholders in the template
            data = self.replace_placeholders(data, {
                "model": self.config["model_name"],
                "prompt": prompt,
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"]
            })
        else:
            # Default to OpenAI format
            data = {
                "model": self.config["model_name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"]
            }
        
        return headers, data
    
    def replace_placeholders(self, template, values):
        """
        Replace placeholders in template with actual values
        
        Args:
            template: Template dictionary or string
            values: Values to replace placeholders
            
        Returns:
            Template with replaced values
        """
        if isinstance(template, dict):
            result = {}
            for key, value in template.items():
                result[key] = self.replace_placeholders(value, values)
            return result
        elif isinstance(template, str):
            for placeholder, value in values.items():
                template = template.replace(f"{{{placeholder}}}", str(value))
            return template
        else:
            return template
    
    def extract_response_text(self, response_json):
        """
        Extract text from API response based on format
        
        Args:
            response_json: JSON response from API
            
        Returns:
            Extracted text or None
        """
        try:
            if self.config["request_format"] == "openai":
                return response_json["choices"][0]["message"]["content"].strip()
            elif self.config["request_format"] == "anthropic":
                return response_json["content"][0]["text"].strip()
            elif self.config["request_format"] == "custom":
                # Use custom extraction path from config
                extraction_path = self.config.get("response_path", ["choices", 0, "message", "content"])
                result = response_json
                for key in extraction_path:
                    if isinstance(key, int):
                        result = result[key]
                    else:
                        result = result[key]
                return result.strip() if isinstance(result, str) else str(result).strip()
            else:
                # Try common paths
                if "choices" in response_json:
                    return response_json["choices"][0]["message"]["content"].strip()
                elif "content" in response_json:
                    return response_json["content"][0]["text"].strip()
                else:
                    logger.warning(f"Unknown response format: {response_json}")
                    return None
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to extract response text: {e}")
            return None
    
    def load_gutenberg_data(self, num_samples=1000):
        """
        Load Gutenberg dataset
        
        Args:
            num_samples: Number of samples to extract
        """
        logger.info("Loading Gutenberg dataset...")
        try:
            # Load Gutenberg dataset
            dataset = load_dataset("sedthh/gutenberg_english", split="train")
            
            # Random sampling and filter texts with appropriate length
            samples = []
            indices = random.sample(range(len(dataset)), min(num_samples * 3, len(dataset)))
            
            for idx in indices:
                text = dataset[idx]['text']
                # Select text paragraphs with moderate length (100-1000 characters)
                if 100 <= len(text) <= 1000:
                    samples.append({
                        'text': text.strip(),
                        'source': 'gutenberg',
                        'label': 0  # Human written
                    })
                    if len(samples) >= num_samples:
                        break
            
            logger.info(f"Extracted {len(samples)} samples from Gutenberg dataset")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load Gutenberg dataset: {e}")
            return []
    
    def load_xsum_data(self, num_samples=1000):
        """
        Load X-Sum dataset
        
        Args:
            num_samples: Number of samples to extract
        """
        logger.info("Loading X-Sum dataset...")
        try:
            # Load X-Sum dataset
            dataset = load_dataset("EdinburghNLP/xsum", split="train")
            
            # Random sampling
            samples = []
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            
            for idx in indices:
                # Use document content instead of summary
                text = dataset[idx]['document']
                # Select texts with appropriate length
                if 100 <= len(text) <= 1000:
                    samples.append({
                        'text': text.strip(),
                        'source': 'xsum',
                        'label': 0  # Human written
                    })
                    if len(samples) >= num_samples:
                        break
            
            logger.info(f"Extracted {len(samples)} samples from X-Sum dataset")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load X-Sum dataset: {e}")
            return []
    
    def call_llm_api(self, text):
        """
        Call LLM API for text rewriting
        
        Args:
            text: Original text
            
        Returns:
            Rewritten text or None (if failed)
        """
        headers, data = self.prepare_api_request(text)
        
        for attempt in range(self.config["max_retries"]):
            try:
                response = requests.post(
                    self.config["api_url"],
                    headers=headers,
                    json=data,
                    timeout=self.config["timeout"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self.extract_response_text(result)
                else:
                    logger.warning(f"API request failed, status code: {response.status_code}")
                    logger.warning(f"Response: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request exception (attempt {attempt + 1}/{self.config['max_retries']}): {e}")
                
            # Wait before retry
            if attempt < self.config["max_retries"] - 1:
                time.sleep(self.config["retry_delay"] * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"API call failed after {self.config['max_retries']} retries")
        return None
    
    def generate_llm_samples(self, human_samples):
        """
        Generate LLM rewritten versions for human samples
        
        Args:
            human_samples: List of human-written samples
            
        Returns:
            List of LLM-generated samples
        """
        logger.info("Generating LLM samples...")
        llm_samples = []
        
        for sample in tqdm(human_samples, desc="Generating LLM samples"):
            rewritten_text = self.call_llm_api(sample['text'])
            
            if rewritten_text:
                llm_samples.append({
                    'text': rewritten_text,
                    'source': sample['source'] + '_llm',
                    'label': 1,  # LLM generated
                    'original_text': sample['text']
                })
            else:
                logger.warning("Skipping one sample, API call failed")
            
            # Control request frequency
            time.sleep(self.config["rate_limit_delay"])
        
        logger.info(f"Generated {len(llm_samples)} LLM samples")
        return llm_samples
    
    def save_to_jsonl(self, samples, output_file):
        """
        Save samples to JSONL format
        
        Args:
            samples: List of samples to save
            output_file: Output file path
        """
        logger.info(f"Saving dataset to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Dataset saved to {output_file}")
    
    def load_from_jsonl(self, input_file):
        """
        Load samples from JSONL format
        
        Args:
            input_file: Input file path
            
        Returns:
            List of samples
        """
        samples = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        
        return samples
    
    def build_dataset(self, gutenberg_samples=500, xsum_samples=500, 
                     output_file="human_vs_llm_dataset.jsonl"):
        """
        Build complete dataset
        
        Args:
            gutenberg_samples: Number of samples to extract from Gutenberg
            xsum_samples: Number of samples to extract from X-Sum
            output_file: Output file name
        """
        logger.info("Starting dataset construction...")
        
        # 1. Load human-written samples
        human_samples = []
        
        # Load Gutenberg samples
        gutenberg_data = self.load_gutenberg_data(gutenberg_samples)
        human_samples.extend(gutenberg_data)
        
        # Load X-Sum samples
        xsum_data = self.load_xsum_data(xsum_samples)
        human_samples.extend(xsum_data)
        
        if not human_samples:
            logger.error("Failed to load any human samples")
            return
        
        logger.info(f"Total loaded {len(human_samples)} human samples")
        
        # 2. Generate LLM samples
        llm_samples = self.generate_llm_samples(human_samples)
        
        # 3. Combine datasets
        all_samples = human_samples + llm_samples
        
        # 4. Shuffle data
        random.shuffle(all_samples)
        
        # 5. Save to JSONL format
        self.save_to_jsonl(all_samples, output_file)
        
        # 6. Print statistics
        self.print_dataset_stats(all_samples)
        
        return all_samples
    
    def print_dataset_stats(self, samples):
        """Print dataset statistics"""
        logger.info("=== Dataset Statistics ===")
        logger.info(f"Total samples: {len(samples)}")
        
        # Count by label
        human_count = sum(1 for s in samples if s['label'] == 0)
        llm_count = sum(1 for s in samples if s['label'] == 1)
        
        logger.info(f"Human-written samples (label=0): {human_count}")
        logger.info(f"LLM-generated samples (label=1): {llm_count}")
        
        # Count by source
        source_counts = {}
        for sample in samples:
            source = sample['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("\nBy source:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count}")
        
        # Text length statistics
        text_lengths = [len(sample['text']) for sample in samples]
        logger.info(f"\nText length statistics:")
        logger.info(f"  Average length: {sum(text_lengths) / len(text_lengths):.1f}")
        logger.info(f"  Minimum length: {min(text_lengths)}")
        logger.info(f"  Maximum length: {max(text_lengths)}")

def main():
    """Main function"""
    # Initialize dataset builder
    builder = UniversalDatasetBuilder("./api_config.json")
    
    # Configuration parameters
    config = {
        'gutenberg_samples': 500,  # Number of Gutenberg samples
        'xsum_samples': 500,       # Number of X-Sum samples
        'output_file': 'human_vs_llm_dataset.jsonl'
    }
    
    # Build dataset
    dataset = builder.build_dataset(**config)
    
    if dataset is not None:
        logger.info("Dataset construction completed!")
        
        # Display first few samples
        print("\n=== Sample Preview ===")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"Source: {sample['source']}")
            print(f"Label: {sample['label']} ({'Human-written' if sample['label'] == 0 else 'LLM-generated'})")
            print(f"Text length: {len(sample['text'])}")
            print(f"Text preview: {sample['text'][:200]}...")
            
            # Show original text for LLM samples
            if sample['label'] == 1 and 'original_text' in sample:
                print(f"Original text preview: {sample['original_text'][:200]}...")
    else:
        logger.error("Dataset construction failed")

# Additional utility functions
def analyze_dataset(jsonl_file):
    """
    Analyze existing JSONL dataset
    
    Args:
        jsonl_file: Path to JSONL dataset file
    """
    builder = UniversalDatasetBuilder()
    samples = builder.load_from_jsonl(jsonl_file)
    builder.print_dataset_stats(samples)
    return samples

def split_dataset(jsonl_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/validation/test sets
    
    Args:
        jsonl_file: Path to JSONL dataset file
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    builder = UniversalDatasetBuilder()
    samples = builder.load_from_jsonl(jsonl_file)
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Calculate split indices
    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split samples
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Save splits
    builder.save_to_jsonl(train_samples, 'train_dataset.jsonl')
    builder.save_to_jsonl(val_samples, 'val_dataset.jsonl')
    builder.save_to_jsonl(test_samples, 'test_dataset.jsonl')
    
    logger.info(f"Dataset split completed:")
    logger.info(f"  Training: {len(train_samples)} samples")
    logger.info(f"  Validation: {len(val_samples)} samples")
    logger.info(f"  Test: {len(test_samples)} samples")

if __name__ == "__main__":
    main()