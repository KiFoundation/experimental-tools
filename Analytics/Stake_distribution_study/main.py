import csv
import time
import psycopg2
import numpy as np
from configparser import ConfigParser


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()

    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    return conn


def get_voting_wallets(end_height, curr):
    query = 'SELECT transactions."serialized", transactions."recipient_id", blocks."height", transactions."asset" ' \
            'FROM transactions INNER JOIN blocks ON blocks."id" = transactions."block_id" ' \
            'WHERE transactions."type" = 3' \
            'AND blocks.height <= ' + str(end_height) + 'ORDER BY blocks."height" ASC;'

    curr.execute(query)
    votes = curr.fetchall()

    return votes


def get_voting_age(end_height, voter_address, curr):
    query = 'SELECT blocks."height" ' \
            'FROM transactions INNER JOIN blocks ON blocks."id" = transactions."block_id" ' \
            'WHERE transactions."type" = 3' \
            ' AND blocks.height <= ' + str(end_height) + \
            ' AND transactions."recipient_id"= ' + str(voter_address) + ' ORDER BY blocks."height" DESC LIMIT 1;'

    curr.execute(query)

    # TODO : FIX THE DEFAULT AGE FOR DELEGATE'S STAKE
    age = curr.fetchall()[0][0] if curr.rowcount != 0 else 0

    return age


def get_registered_delegates_to_height(height_, curr):
    query = 'SELECT transactions."sender_public_key"' \
            'FROM transactions INNER JOIN blocks ON blocks."id" = transactions."block_id" ' \
            'WHERE blocks."height" <= ' + str(height_) + \
            ' AND transactions."type"= 2;'

    curr.execute(query)
    delegates = curr.fetchall()

    return [delegate[0] for delegate in delegates]


def get_transaction_for_voting_wallets(end_height, voters_public_keys, voters_addresses, curr):
    txs = []

    time_tx = time.time()

    for voter_address in np.unique(voters_addresses):
        query = 'SELECT transactions."serialized", transactions."amount", transactions."recipient_id",' \
                'transactions."sender_public_key", transactions."fee", transactions."vendor_field_hex",' \
                'transactions."timestamp", blocks."height" ,  transactions."id"' \
                ' FROM transactions INNER JOIN blocks ON blocks."id" = transactions."block_id"  ' \
                ' WHERE blocks."height" <= ' + str(end_height) + \
                ' AND transactions."recipient_id"= ' + str(voter_address) + \
                ' ORDER BY blocks."height" ASC;'

        curr.execute(query)
        txs = txs + curr.fetchall()

    for voter_public_key in np.unique(voters_public_keys):
        query = 'SELECT transactions."serialized", transactions."amount", transactions."recipient_id",' \
                'transactions."sender_public_key", transactions."fee", transactions."vendor_field_hex",' \
                'transactions."timestamp", blocks."height",  transactions."id"' \
                ' FROM transactions INNER JOIN blocks ON blocks."id" = transactions."block_id"  ' \
                ' WHERE blocks."height" <= ' + str(end_height) + \
                ' AND transactions."sender_public_key" =' + str(voter_public_key) + \
                ' ORDER BY blocks."height" ASC;'

        curr.execute(query)
        txs = txs + curr.fetchall()

    # print(txs)
    print('Time for tx: ', time.time() - time_tx)
    return txs


def get_wallet_addresses_db(ids, curr):
    res_ = []
    ids_ = []
    start_db = time.time()
    for id in ids:
        cur.execute("SELECT sender_public_key "
                    "FROM transactions "
                    "WHERE type = 3 AND recipient_id =" + "'" + str(id) + "' LIMIT 1")

        res_.append(curr.fetchall()[0][0])
        ids_.append(id)

    print("Time for DB calls : ", time.time() - start_db)
    return ids_, res_


def get_delegate_addresses_file(ids, file):
    data = {}
    res_ = []
    ids_ = []

    start_api = time.time()

    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data[row[1]] = row[0]

    for id in ids:
        res_.append(data[id])
        ids_.append(id)

    print("Time for local calls : ", time.time() - start_api)
    return ids_, res_


def get_block_rewards(delegate, curr, height_):
    query = 'SELECT sum(total_fee) + sum(reward) FROM blocks ' \
            'WHERE height <= ' + str(height_) + ' AND generator_public_key = ' + "'" + str(delegate) + "'" + ' ;'

    curr.execute(query)
    block_reward = curr.fetchall()[0][0]

    return block_reward if block_reward is not None else 0


def update_wallet_balances(voter_ids, voter_keys, delegates_ids, delegates_keys, transactions):
    voter_wallets_keys_to_ids = dict(zip(voter_keys + delegates_keys, voter_ids + delegates_ids))
    balances_ids = dict(zip(voter_ids + delegates_ids, [0 for i in range(len(voter_keys + voter_ids))]))

    for delegate_key in delegates_keys:
        reward = float(get_block_rewards(delegate_key, cur, height)) / 100000000
        balances_ids[voter_wallets_keys_to_ids[delegate_key]] += reward

    start_bal = time.time()
    transactions_sorted = sorted(transactions, key=lambda i: i[7])

    for transaction in transactions_sorted:
        amount = transaction[1]
        recipient = transaction[2]
        sender = transaction[3]
        fee = transaction[4]

        if recipient in voter_ids:
            balances_ids[recipient] += amount / 100000000

        if sender in voter_keys:
            balances_ids[voter_wallets_keys_to_ids[sender]] -= amount / 100000000 + fee / 100000000

    print("Time for balance update : ", time.time() - start_bal)

    return balances_ids


def write_voting_wallet_balances(height_, curr):
    start = time.time()

    # get the list of voters
    records_voters = get_voting_wallets(height_, curr)

    # get the list of wallet keys from the DB
    ids, keys = get_wallet_addresses_db([record[1] for record in records_voters], curr)

    # get the list of registered delegates up to the height
    delegates = get_registered_delegates_to_height(height_, curr)
    keys_del, ids_del = get_delegate_addresses_file(delegates, 'data/delegates.csv')

    # get the list of transaction related to the voter wallets
    records_transactions = get_transaction_for_voting_wallets(height_, ["'" + key + "'" for key in keys],
                                                              ["'" + id + "'" for id in ids], curr)

    records_transactions_dict = {}
    for records_transaction in records_transactions:
        records_transactions_dict[records_transaction[8]] = records_transaction

    records_transactions_unique = records_transactions_dict.values()

    # update the voter wallet balances
    balances = update_wallet_balances(ids, keys, ids_del, keys_del, records_transactions_unique)
    print('# Votes : ', len(records_voters))
    print('# Voters : ', len(balances))
    print('# Transactions : ', len(records_transactions))

    end = time.time()
    print('Total time: ', end - start)

    measures_ = str(height_) + ',' \
                + str(len(records_voters)) + ',' \
                + str(len(records_transactions)) + ',' \
                + str(len(balances)) + ',' \
                + str(max(balances.values())) + ',' \
                + str(end - start) + '\n'

    f = open('output/' + str(height_) + '.csv', 'w')
    f.write("key, balance, age\n")

    for balance in balances.keys():
        age = get_voting_age(height_, "'" + balance + "'", curr)
        f.write(str(balance) + ',' + str(round(balances[balance], 5)) + ',' + str(age) + '\n')

    f.close()

    return measures_


def write_delegate_stakes(height_, curr):
    start = time.time()

    # get the list of voters
    records_voters_ = get_voting_wallets(height_, curr)

    # get balances at height from file
    balances_ = {}

    with open('output/' + str(height_) + '.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            balances_[row[0]] = float(row[1])

    # get delegates stake for this height
    votes = [[record[1], record[3]['votes'][0].strip('+-')] for record in records_voters_]
    delegates_voted = np.unique([vote[1] for vote in votes])
    delegates_stakes = dict(zip(delegates_voted, [0 for i in delegates_voted]))
    delegates_stake_string = 'delegate,stake\n'

    for vote in votes:
        delegates_stakes[vote[1]] += balances_[vote[0]]

    # get the list of registered delegates up to the height
    delegates = get_registered_delegates_to_height(height_, curr)
    keys_del, ids_del = get_delegate_addresses_file(delegates, 'data/delegates.csv')
    keys_to_ids = dict(zip(keys_del, ids_del))

    for delegate_stake in delegates_stakes.keys():
        delegates_stakes[delegate_stake] += balances_[keys_to_ids[delegate_stake]]
        delegates_stake_string += delegate_stake + ',' + str(delegates_stakes[delegate_stake]) + '\n'

    f_ = open('output_del/' + str(height_) + '.csv', 'w')
    f_.write(delegates_stake_string)
    f_.close()

    print('Total time: ', time.time() - start)

    return


if __name__ == '__main__':
    con = connect()
    cur = con.cursor()
    heights = range(1000000, 5100000, 500000)
    del_stakes = 1
    vot_balances = 0

    if vot_balances:
        measures = 'height, votes, records_transactions, voters, max, time\n'
        for height in heights:
            print(height)
            measures += write_voting_wallet_balances(height, cur)

        f1 = open('output/logs' + str(heights[0]) + '.csv', 'w')
        f1.write(measures)
        f1.close()

    if del_stakes:
        for height in heights:
            write_delegate_stakes(height, cur)

    cur.close()
    con.close()
