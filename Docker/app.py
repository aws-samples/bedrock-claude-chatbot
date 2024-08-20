import json
import traceback
import subprocess
import base64
import os
import boto3
import sys
import re

class CodeExecutionError(Exception):
    pass

def local_code_executy(code_string):
    """
    Execute a given Python code string in a secure, isolated environment and capture its output.

    Parameters:
    code_string (str): The Python code to be executed.

    Returns:
    dict: The output of the executed code, expected to be in JSON format.

    Raises:
    CodeExecutionError: Custom exception with detailed error information if code execution fails.

    
    Functionality:
    1. Creates a temporary Python file with a unique name in the /tmp directory.
    2. Writes the provided code to this temporary file.
    3. Executes the temporary file using the Python interpreter.
    4. Captures the output from a predefined output file ('/tmp/output.json').
    5. Cleans up temporary files after execution.
    6. In case of execution errors, provides detailed error information including:
       - The error message and traceback.
       - The line number where the error occurred.
       - Context of the code around the error line.

    Note:
    - This function assumes that the executed code writes its output to '/tmp/output.json'. This (saving the output to local) is appended to the generated code in the main application.
    """
    # Create a unique filename in /tmp
    temp_file_path = f"/tmp/code_{os.urandom(16).hex()}.py"
    output_file_path = '/tmp/output.json'
    try:
        # Write the code to the temporary file
        with open(temp_file_path, 'w', encoding="utf-8") as temp_file:
            temp_file.write(code_string)
        
        # Execute the temporary file
        result = subprocess.run([sys.executable, temp_file_path], 
                                capture_output=True, text=True, check=True)
       
        with open(output_file_path, 'r', encoding="utf-8") as f:
            output = json.load(f)

        # Clean up temporary files
        os.remove(output_file_path)

        return output

    except subprocess.CalledProcessError as e:
        # An error occurred during execution
        full_error_message = e.stderr.strip()

        # Extract the traceback part of the error message
        traceback_match = re.search(r'(Traceback[\s\S]*)', full_error_message)
        if traceback_match:
            error_message = traceback_match.group(1)
        else:
            error_message = full_error_message  # Fallback to full message if no traceback found 

        # Parse the traceback to get the line number
        tb_lines = error_message.split('\n')
        line_no = None
        for line in reversed(tb_lines):
            if temp_file_path in line:
                match = re.search(r'line (\d+)', line)
                if match:
                    line_no = int(match.group(1))
                break

        # Construct error message with context
        error = f"Error: {error_message}\n"
        if line_no is not None:
            code_lines = code_string.split('\n')
            context_lines = 2
            start = max(0, line_no - 1 - context_lines)
            end = min(len(code_lines), line_no + context_lines)
            error += f"Error on line {line_no}:\n"
            for i, line in enumerate(code_lines[start:end], start=start+1):
                prefix = "-> " if i == line_no else "   "
                error += f"{prefix}{i}: {line}\n"
        else:
            error += "Could not determine the exact line of the error.\n"
            error += "Full code:\n"
            for i, line in enumerate(code_string.split('\n'), start=1):
                error += f"{i}: {line}\n"

        raise CodeExecutionError(error)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



def execute_function_string(input_code, trial, bucket, key_prefix):
    """
    Execute a given Python code string, potentially modifying dataset paths to use S3. 
    If it's the first trial (trial < 1) and the S3 bucket/prefix are not already in the code:
       - Replaces local dataset references with S3 URIs.

    Parameters:
    input_code (dict): A dictionary containing the following keys:
        - 'code' (str): The Python code to be executed.
        - 'dataset_name' (str or list, optional): Name(s) of the dataset(s) used in the code.
    trial (int): A counter for execution attempts, used to determine if S3 paths should be injected.
    bucket (str): The name of the S3 bucket where datasets are stored.
    key_prefix (str): The S3 key prefix (folder path) where datasets are located within the bucket.

    Returns:
    The result of executing the code using the local_code_executy function.

    """
    code_string = input_code['code']
    dataset_names = input_code.get('dataset_name', [])
    if isinstance(dataset_names, str):
        dataset_names = [d.strip() for d in dataset_names.strip('[]').split(',')]

    #BUCKET = os.environ.get('BUCKET', '')
    #S3_DOC_CACHE_PATH = os.environ.get('S3_DOC_CACHE_PATH', '')

    if trial < 1 and (key_prefix or bucket) not in code_string:
        for dataset_name in dataset_names:
            code_string = code_string.replace(dataset_name, f"s3://{bucket}/{key_prefix}/{dataset_name}")    
    return local_code_executy(code_string)


def put_obj_in_s3_bucket_(docs, bucket, key_prefix):
    """Uploads a file to an S3 bucket and returns the S3 URI of the uploaded object.
    Args:
       docs (str): The local file path of the file to upload to S3.
       bucket (str): S3 bucket name,
       key_prefix (str): S3 key prefix.
   Returns:
       str: The S3 URI of the uploaded object, in the format "s3://{bucket_name}/{file_path}".
    """
    S3 = boto3.client('s3')
    if isinstance(docs,str):
        file_name=os.path.basename(docs)
        file_path=f"{key_prefix}/{docs}"
        S3.upload_file(f"/tmp/{docs}", bucket, file_path)
    else:
        file_name=os.path.basename(docs.name)
        file_path=f"{key_prefix}/{file_name}"
        S3.put_object(Body=docs.read(),Bucket= BUCKET, Key=file_path)           
    return f"s3://{bucket}/{file_path}"



def lambda_handler(event, context):
    try:
        input_data = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        iterate = input_data.get('iterate', 0)
        bucket=input_data.get('bucket','')
        s3_file_path=input_data.get('file_path','')
        result = execute_function_string(input_data, iterate, bucket, s3_file_path)
        print(result)
        image_holder = []

        if isinstance(result, dict):
            for item, value in result.items():
                if "image" in item and value is not None:
                    if isinstance(value, list):
                        for img in value:
                            image_path_s3 = put_obj_in_s3_bucket_(img, bucket, s3_file_path)
                            image_holder.append(image_path_s3)                            
                    else:                        
                        image_path_s3 = put_obj_in_s3_bucket_(value,bucket,s3_file_path)
                        image_holder.append(image_path_s3)

        tool_result = {
            "result": result,            
            "image_dict": image_holder
        }

        return {
            'statusCode': 200,
            'body': json.dumps(tool_result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }