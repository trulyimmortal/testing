import requests
import json

# Test script for the API
def test_api(base_url="http://localhost:8000"):
    """Test the QuestionCraft AI API"""
    
    print(f"Testing API at: {base_url}")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test question generation with marketing content
    marketing_content = """
    The Marketing Mix, also known as the 4Ps of Marketing, is a fundamental concept in marketing strategy. 
    It consists of Product, Price, Place, and Promotion. The Product refers to the goods or services offered 
    to customers. Price is the amount customers pay for the product. Place involves the distribution channels 
    and locations where the product is available. Promotion encompasses all marketing communications used to 
    inform and persuade customers. Companies must carefully balance these four elements to create an effective 
    marketing strategy that meets customer needs while achieving business objectives.
    """
    
    test_cases = [
        {
            "name": "Marketing Mix",
            "content": marketing_content
        },
        {
            "name": "Property Rights",
            "content": """
            Individual property rights are fundamental legal concepts that define ownership and control over assets. 
            These rights include the right to use, transfer, and exclude others from property. Property can be 
            tangible, such as real estate and personal belongings, or intangible, such as intellectual property 
            and financial assets. The legal framework governing property rights varies by jurisdiction but 
            generally provides protection against unauthorized use or seizure. Understanding property rights 
            is essential for economic transactions, investment decisions, and legal compliance.
            """
        },
        {
            "name": "Machine Learning",
            "content": """
            Machine Learning is a subset of artificial intelligence that enables computers to learn and improve 
            from experience without being explicitly programmed. It involves algorithms that can identify patterns 
            in data and make predictions or decisions. There are three main types: supervised learning, 
            unsupervised learning, and reinforcement learning. Applications include image recognition, 
            natural language processing, recommendation systems, and autonomous vehicles.
            """
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        payload = {
            "content": test_case["content"],
            "max_questions": 8
        }
        
        try:
            response = requests.post(
                f"{base_url}/generate-questions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Generated {result['total_generated']} questions in {result['processing_time']}s:")
                for i, question in enumerate(result['questions'], 1):
                    print(f"{i}. {question}")
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    # Test locally
    test_api("http://localhost:8000")
    
    # Uncomment to test deployed version
    # test_api("https://your-app-name.railway.app")
    # test_api("https://your-app-name.onrender.com")
