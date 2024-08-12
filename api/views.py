from django.http import HttpResponse
import json
import pickle
import base64
import pandas as pd
import csv
from supabase import create_client, Client
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from .train_ai import AI
import traceback
from datetime import datetime, timedelta, timezone
from django.utils.crypto import get_random_string

from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
import requests

@csrf_exempt
def validate_token(token):
    try:
        url = settings.SUPABASE_URL
        auth_url = f"{url}/auth/v1/user"
        key = settings.SUPABASE_KEY

        headers = {
            'Authorization': f'Bearer {token}',
            'apikey': key
        }
        response = requests.get(auth_url, headers=headers)

        if response.status_code == 200:
            user_data = response.json()
            user_id = user_data['id']  # Adjust this if the structure is different
            return user_id
        else:
            raise Exception(f"Token validation failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        raise Exception(f"Token validation failed: {str(e)}")


@csrf_exempt
def send_verification_email(user_email, token):
    endpoint = "verify"
    domain = settings.MAILGUN_DOMAIN
    mailgun_api = settings.MAILGUN_API
    return requests.post(
        f"{domain}",
        auth=("api", f"{mailgun_api}"),
        data={"from": f"Excited User <mailgun@sandboxcdbca9366452406386a436563157c9aa.mailgun.org>",
              "to": [user_email],
              "subject": "Verify your email",
              "html": f"""<html>
                          <body>
                            <p>To Go http://127.0.0.1:8000/{endpoint}/{token}/ to verify your email.</p>
                          </body>
                        </html>"""
        }
    )

@csrf_exempt
def signup(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    auth_url = f"{url}/auth/v1"
    data = json.loads(request.body)
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    auth_response = requests.post(
         f"{auth_url}/signup",
        json={"email": email, "password": password},
        headers={"apikey": key, "Content-Type": "application/json"}
    )

    if auth_response.status_code == 200:
        user_data = auth_response.json()
            
        # Check the structure of user_data from Supabase response
        if 'user' in user_data and 'id' in user_data['user']:
            user_id = user_data['user']['id']
        else:
            return JsonResponse({"error": "User ID not found in response"}, status=status.HTTP_400_BAD_REQUEST)

        # Insert user data into Users table
        token = get_random_string(length=32)
        send_verification_email(email, token)
        supabase = create_client(url, key)
        supabase.table('Users').insert({
            'user_id': user_id,
            'name': name,
            'email': email, 
            'token': str(token)
        }).execute()

        return JsonResponse({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
    else:
        print(traceback.format_exc())
        return JsonResponse(auth_response.json(), status=auth_response.status_code)


@csrf_exempt
def login(request):
    # Fetching Supabase URL and Key from settings
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY
    auth_url = f"{url}/auth/v1"

    # Extract email and password from request data
    data = json.loads(request.body)
    email = data.get('email')
    password = data.get('password')

    # Supabase authentication request
    try:
        auth_response = requests.post(
            f"{auth_url}/token?grant_type=password",
            json={"email": email, "password": password},
            headers={"apikey": key, "Content-Type": "application/json"}
        )
        auth_response.raise_for_status()
    except requests.RequestException as e:
        print(f"Authentication request failed: {e}")
        return JsonResponse({'error': 'Authentication request failed'}, status=500)


    if auth_response.status_code == 200:
        # Supabase client creation
        supabase: Client = create_client(url, key)
        
        # Fetch user details from Supabase
        user_response = supabase.table('Users').select('*').eq('email', email).execute()

        if user_response.data:
            user = user_response.data[0]

            if user['status'] == 'verified':
                created_at = datetime.fromisoformat(user['created_at']).astimezone(timezone.utc)
                current_time = datetime.now(timezone.utc)
                
                if current_time < created_at + timedelta(days=14) or user['plan'] == 'paid':
                    tokens = auth_response.json()
                    return JsonResponse(tokens)
                else: 
                    return JsonResponse({'pay': 'need payment'}, status=402)
            else: 
                return JsonResponse({'verify': 'need verify'}, status=403)
        else:
            return JsonResponse({'error': 'User not found'}, status=404)
    else:
        # Log the authentication error details (for debugging purposes)
        print(f"Auth error: {auth_response.json()}")
        return JsonResponse(auth_response.json(), status=auth_response.status_code)
    
        
@csrf_exempt
def verify(request, token):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase = create_client(url, key)

    if request.method == 'GET':
        # Find the user with the given token
        try:
            user = supabase.table('Users').select('*').eq('token', token).execute().data
            
            if not user:
                return JsonResponse({'error': 'Invalid or expired token'}, status=400)
            
            # Insert the user into the main collection   
            supabase.table('Users').update({'status': 'verified'}).eq('token', token).execute()
            
            return JsonResponse({'message': 'Email verified successfully. You can now log in.'}, status=200)
        except Exception as e:
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=200)
            
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    

        

# Create your views here.
def main(req):
    return HttpResponse("Wsg")



@csrf_exempt
def user_info(req):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase = create_client(url, key)

    try:
        token = req.headers.get('Authorization', '').split('Bearer ')[-1]
        try:
            user_id = validate_token(token)
        except Exception as e:
            return JsonResponse({'auth': True}, status=403)
        user = supabase.table('Users').select('*').eq('user_id', user_id).execute().data

        return JsonResponse({'user': user})

    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def get_prediction(req):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase = create_client(url, key)

    try:
        data = json.loads(req.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    try:
        token = req.headers.get('Authorization', '').split('Bearer ')[-1]
        try:
            user_id = validate_token(token)
        except Exception as e:
            return JsonResponse({'auth': True}, status=403)

        input_df = pd.DataFrame([data])

        model_data = supabase.table('Models').select('*').eq('user_id', user_id).eq('name', input_df.get('model', [''])[0]).execute().data[0]
        model = pickle.loads(base64.b64decode(model_data['model']))
        feature_columns = pickle.loads(base64.b64decode(model_data['feature_columns']))
        target_columns = pickle.loads(base64.b64decode(model_data['target_columns']))

        # Drop the 'model' column if present
        input_df = input_df.drop('model', axis=1, errors='ignore')

        # Ensure all feature columns are present
        missing_cols = [col for col in feature_columns if col not in input_df.columns]
        for col in missing_cols:
            input_df[col] = 0

        # Convert to dummy variables
        input_df_encoded = pd.get_dummies(input_df, columns=[col for col in feature_columns if col in input_df.columns], drop_first=False)

        # Reorder columns to match the training set
        input_df_encoded = input_df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Convert to numpy array
        x_input = input_df_encoded.to_numpy()

        # Make the prediction
        probabilities = model.predict(x_input)
        predicted_class = np.argmax(probabilities, axis=1)
        prediction_probability = float(probabilities[0][predicted_class[0]])

        # Map predictions back to original classes
        encoded_result = target_columns[predicted_class[0]]

        return JsonResponse({'result': encoded_result, 'prob': prediction_probability})

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except KeyError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


def find_features(req):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase: Client = create_client(url, key)

    token = req.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    try:
        # Retrieve data from "AI_data" table
        data = supabase.table('AI_Data').select('*').eq('user_id', user_id).execute()

        number_of_rows = len(data.data)
        if number_of_rows < 20:
            return JsonResponse({'warning': 'Need 20 rows of data'})

        # Convert data to DataFrame
        df = pd.DataFrame(data.data)

        # Drop unnecessary columns
        df = df.drop(columns=['id', 'created_at', 'user_id'])

        # Instantiate AI class and preprocess data
        ai = AI()
        x_values, y_values = ai.preprocess_data(df)

        # Train the neural network model
        num_nodes = 12
        epochs = 20
        batch_size = 16
        lr = 0.005
        dropout_prob = 0
        model, accuracy, shap_df = ai.create_nn_model(x_values, y_values, num_nodes, epochs, batch_size, lr, dropout_prob)
        feature_importance_json = shap_df.to_json(orient='records')
        
        return JsonResponse({'message': feature_importance_json})

    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse({'error': 'Failed to process data', 'details': str(e)}, status=500)


@csrf_exempt
def store_model(req):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase: Client = create_client(url, key)

    token = req.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    # Retrieve data from "AI_data" table
    data = supabase.table('AI_Data').select('*').eq('user_id', user_id).execute()

    number_of_rows = len(data.data)
    if number_of_rows < 20:
        return JsonResponse({'message': 'Need 20 rows of data'})

    # Convert data to DataFrame
    df = pd.DataFrame(data.data)

    # Drop unnecessary columns
    df = df.drop(columns=['id', 'created_at', 'user_id'])

    # Instantiate AI class and preprocess data
    ai = AI()
    x_values, y_values = ai.preprocess_data(df)

    # Train the neural network model
    num_nodes = 64
    epochs = 50
    batch_size = 32
    lr = 0.005
    dropout_prob = 0
    model, accuracy, shap_df = ai.create_nn_model(x_values, y_values, num_nodes, epochs, batch_size, lr, dropout_prob)

    # Serialize the model and save to Supabase
    model_serialized = base64.b64encode(pickle.dumps(model)).decode('utf-8')
    feature_columns_serialized = base64.b64encode(pickle.dumps(ai.feature_columns)).decode('utf-8')
    target_columns_serialized = base64.b64encode(pickle.dumps(ai.target_columns)).decode('utf-8')

    try:
        data = json.loads(req.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    # Save to Supabase
    model_data = {
        'user_id': user_id,
        'name': data.get('name', ''),
        'model': model_serialized,
        'feature_columns': feature_columns_serialized,
        'target_columns': target_columns_serialized,
        'created_at': datetime.now().isoformat(),
        'size': number_of_rows, 
        'accuracy': str(accuracy*100) + "%"
    }
    supabase.table('Models').insert(model_data).execute()

    return JsonResponse({'message': "success"})


@csrf_exempt
def insert_row(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        required_fields = [
            'response', 'location', 'lead_gen', 
            'offer', 'sale_product', 'funnel_stage'
        ]
        for field in required_fields:
            if field not in data:
                return JsonResponse({'error': f'The field "{field}" is required.'}, status=400)

        try:
            data['user_id'] = user_id
            # Insert data into the AI_Data table in Supabase
            response = supabase.table('AI_Data').insert([data]).execute()

            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)



@csrf_exempt
def delete_row(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    data = json.loads(request.body)
    row = data.get('row')

    if not row:
        return JsonResponse({'error': 'row are required'}, status=400)

    try:
        response = supabase.table('AI_Data').delete().match({'id': row, 'user_id': user_id}).execute()
        return JsonResponse({'message': 'Entry deleted successfully'}, status=200)
    except:
        print(traceback.format_exc())
        return JsonResponse({'error': 'Failed to delete entry'})


@csrf_exempt
def add_feature(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    req_data = json.loads(request.body)

    token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    table = req_data.get('table')
    text = req_data.get('text')

    if not table or not text:
        return JsonResponse({'error': 'Table and text are required'}, status=400)
    
    data = {
        'user_id': user_id,
        'main': text
    }
    
    response = supabase.table(table).insert(data).execute()

    return JsonResponse({'message': 'Entry added successfully'}, status=201)



@csrf_exempt
def delete_feature(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    table = request.POST.get('feature')
    text = request.POST.get('option')

    if not table or not text:
        return JsonResponse({'error': 'Feature and option are required'}, status=400)

    try:
        response = supabase.table(table).delete().match({'main': text, 'user_id': user_id}).execute()
        return JsonResponse({'message': 'Entry deleted successfully'}, status=200)
    except:
        return JsonResponse({'error': 'Failed to delete entry'}, status=response.status_code)
    


@csrf_exempt
def edit_feature(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    try:
        user_id = validate_token(token)
    except Exception as e:
        return JsonResponse({'auth': True}, status=403)

    req_data = json.loads(request.body)

    table = req_data.get('feature')
    data_table = (table.lower())[:-1]
    main = req_data.get('main')
    text = req_data.get('text')

    if not table or not main:
        return JsonResponse({'error': 'Feature and option are required'}, status=400)

    try:
        supabase.table(table).update({'main': text}).eq('main', main).eq('user_id', user_id).execute()
        supabase.table('AI_Data').update({data_table: text}).eq(data_table, main).eq('user_id', user_id).execute()
        return JsonResponse({'success': True})
    except:
        print(traceback.format_exc())
        return JsonResponse({'error': 'Failed to edit entry'})
        


@csrf_exempt
def get_feature_values(req):
    def fetch_table_data(supabase, table_name, user_id):
        try:
            response = supabase.table(table_name).select("*").eq('user_id', user_id).execute()
            return response.data
        except Exception as e:
            raise Exception(f"Error fetching data from {table_name}: {str(e)}")

    if req.method != "GET":
        return HttpResponse(status=405)  # Method Not Allowed

    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase: Client = create_client(url, key)

    try:
        token = req.headers.get('Authorization', '').split('Bearer ')[-1]
        try:
            user_id = validate_token(token)
        except Exception as e:
            return JsonResponse({'auth': True}, status=403)

        tables = ["Responses", "Locations", "Offers", "Sale_Products", "Lead_Gens", "Results", "Funnel_Stages"]
        dropdown_options = {}

        for table in tables:
            data = fetch_table_data(supabase, table, user_id)
            dropdown_options[table] = [{"main": item["main"], "id": item["id"]} for item in data]

        return JsonResponse(dropdown_options)
    
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return JsonResponse({"error": str(e)}, status=500)
    

@csrf_exempt
def get_feature_values_and_models(req):
    def fetch_table_data(supabase, table_name, user_id):
        try:
            response = supabase.table(table_name).select("*").eq('user_id', user_id).execute()
            return response.data
        except Exception as e:
            raise Exception(f"Error fetching data from {table_name}: {str(e)}")

    if req.method != "GET":
        return HttpResponse(status=405)  # Method Not Allowed

    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Connect to Supabase
    supabase: Client = create_client(url, key)

    try:
        token = req.headers.get('Authorization', '').split('Bearer ')[-1]
        try:
            user_id = validate_token(token)
        except Exception as e:
            return JsonResponse({'auth': True}, status=403)

        tables = ["Responses", "Locations", "Offers", "Sale_Products", "Lead_Gens", "Results", "Funnel_Stages"]
        dropdown_options = {}

        for table in tables:
            data = fetch_table_data(supabase, table, user_id)
            dropdown_options[table] = [{"main": item["main"], "id": item["id"]} for item in data]

        # Fetch models
        model_response = supabase.table('Models').select("*").eq('user_id', user_id).order('created_at', desc=True).limit(5).execute()
        model_data = model_response.data
        dropdown_options['models'] = [{"main": item["name"], "id": item["id"]} for item in model_data]

        return JsonResponse(dropdown_options)
    
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return JsonResponse({"error": str(e)}, status=500)



def recent_data(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    if request.method == "GET":
        try:
            token = request.headers.get('Authorization', '').split('Bearer ')[-1]
            try:
                user_id = validate_token(token)
            except Exception as e:
                return JsonResponse({'auth': True}, status=403)
            
            # Fetch data from the AI_Data table in Supabase and sort by a timestamp column
            data = supabase.table('AI_Data').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(3).execute()

            if data.data:
                return JsonResponse({'recent_data': data.data})
            else:
                return JsonResponse({'recent_data': []})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def data(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    if request.method == "GET":
        try:
            token = request.headers.get('Authorization', '').split('Bearer ')[-1]
            try:
                user_id = validate_token(token)
            except Exception as e:
                return JsonResponse({'auth': True}, status=403)
            # Fetch data from the AI_Data table in Supabase and sort by a timestamp column
            data = supabase.table('AI_Data').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()

            if data.data:
                return JsonResponse({'data': data.data})
            else:
                return JsonResponse({'data': []})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def recent_models(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    if request.method == "GET":
        try:
            token = request.headers.get('Authorization', '').split('Bearer ')[-1]
            try:
                user_id = validate_token(token)
            except Exception as e:
                return JsonResponse({'auth': True}, status=403)
            # Fetch data from the AI_Data table in Supabase and sort by a timestamp column
            data = supabase.table('Models').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(2).execute()

            if data.data:
                return JsonResponse({'recent_models': data.data})
            else:
                return JsonResponse({'recent_models': []})
        except Exception as e:
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)
            

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def models(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    if request.method == "GET":
        try:
            token = request.headers.get('Authorization', '').split('Bearer ')[-1]
            try:
                user_id = validate_token(token)
            except Exception as e:
                return JsonResponse({'auth': True}, status=403)
            # Fetch data from the AI_Data table in Supabase and sort by a timestamp column
            data = supabase.table('Models').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()

            if data.data:
                return JsonResponse({'models': data.data})
            else:
                return JsonResponse({'models': []})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def export_csv(request):
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY

    # Initialize Supabase client
    supabase: Client = create_client(url, key)

    # Fetch data from Supabase
    try:
        token = request.headers.get('Authorization', '').split('Bearer ')[-1]
        try:
            user_id = validate_token(token)
        except Exception as e:
            return JsonResponse({'auth': True}, status=403)

        data = supabase.table('AI_Data').select('*').eq('user_id', user_id).order('created_at', desc=True).execute().data
        
        data = [{k: v for k, v in row.items() if k not in ['id', 'user_id']} for row in data]
        
        # Create the HttpResponse object with the appropriate CSV header.
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="salesight_data.csv"'

        # Create a CSV writer
        writer = csv.writer(response)

        # Write the headers if data is not empty
        if data:
            headers = data[0].keys()
            writer.writerow(headers)

            # Write the data rows
            for row in data:
                writer.writerow(row.values())

        return response
    except Exception as e:
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)