import time
import requests


# Get wallet keys from wallet addresses
def get_wallet_keys_api(ids):
    res_ = []
    ids_ = []
    start_api = time.time()
    print('started')
    for id in ids:
        r = requests.get('https://explorer.ark.io/api/v2/wallets/' + id)
        res_.append(r.json()['data']['publicKey'])
        ids_.append(id)
    print("Time for API calls : ", time.time() - start_api)
    return ids_, res_


# Get wallet addresses from wallet keys
def get_wallet_ids_api(keys):
    res_ = []
    keys_ = []
    start_api = time.time()
    for key in keys:
        r = requests.get('https://explorer.ark.io/api/v2/wallets/' + key)
        res_.append(r.json()['data']['address'])
        keys_.append(key)

    print("Time for API calls : ", time.time() - start_api)
    return keys_, res_


# Get all registered delegates from API
def get_all_delegates_keys_api():
    res_ = []
    ids_ = []
    start_api = time.time()
    print('started')
    r = requests.get('https://explorer.ark.io/api/v2/delegates')
    pages = r.json()['meta']['pageCount']

    for page in range(1, pages + 1, 1):
        r = requests.get('https://explorer.ark.io/api/v2/delegates?page=' + str(page))
        for ele in r.json()['data']:
            ids_.append(ele['address'])
            res_.append(ele['publicKey'])

    print("Time for API calls : ", time.time() - start_api)
    return ids_, res_


# get delegate names from ids
def get_delegate_names(ids):
    res = {}
    for id_ in ids:
        r = requests.get('https://explorer.ark.io/api/v2/delegates/' + id_)
        res[id_] = r.json()['data']['username']
    return res


# Get the balance of a wallet at a given height
def compute_wallet_balance_at_height_api(key_, height_):
    # get height timestamp
    r = requests.get('https://explorer.ark.io/api/v2/blocks?height=' + str(height_))

    if r.status_code == 404:
        print("The block does not exist")
        return

    timestamp = r.json()['data'][0]['timestamp']['unix']

    # get transaction page counts
    r = requests.get('https://explorer.ark.io/api/wallets/' + key_ + '/transactions')
    pages = r.json()['meta']['pageCount']

    # parse the pages and iteratively compute the balance
    sent = 0
    received = 0

    if r.status_code == 404:
        print("The wallet does not exist")
        return

    for page in range(1, pages + 1, 1):
        r = requests.get(
            'https://explorer.ark.io/api/wallets/' + key_ + '/transactions?page=' + str(page))

        for ele in r.json()['data']:
            # compute sent amount
            if ele['timestamp']['unix'] < timestamp and ele['sender'] == key_:
                sent += int(ele['amount'])
                sent += int(ele['fee'])

            # compute received amount
            if ele['timestamp']['unix'] < timestamp and ele['recipient'] == key_:
                received += int(ele['amount'])

    total_tx = received - sent
    print("The total spent tokens :", sent / 100000000)
    print("The total received tokens :", received / 100000000)
    print("Transaction balance :", total_tx / 100000000)

    # Get forged tokens
    r = requests.get('https://explorer.ark.io/api/v2/delegates/' + key_ + '/blocks')
    total_forged = 0

    if r.status_code == 404:
        print("The wallet is not regostered as delegate")
    else:
        # parse the pages and iteratively compute the number of forged tokens
        pages = r.json()['meta']['pageCount']
        for page in range(1, pages + 1, 1):
            r = requests.get('https://explorer.ark.io/api/v2/delegates/' + key_ + '/blocks?page=' + str(page))
            for ele in r.json()['data']:
                if ele['timestamp']['unix'] < timestamp:
                    total_forged += int(ele['forged']['total'])

    print("The total forged tokens :", total_forged / 100000000)
    print("The balance at height {height} is {balance}".format(height=height_,
                                                               balance=(total_tx + total_forged) / 100000000))

    return
