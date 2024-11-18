
import configparser
from neo4j import GraphDatabase


def clear_db(tx):
    tx.run('MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r')


def search_all(tx):
    result = tx.run('MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN n,r')
    return [r for r in result]


def add_person_node(tx, name):
    tx.run('CREATE (p:Person {name: $name}) RETURN p', {'name': name})


def add_friend_relationship(tx, name, friend_name=None):
    if not friend_name:
        tx.run('CREATE (p:Person {name: $name}) RETURN p', {'name': name})
    else:
        tx.run('MATCH (p:Person {name: $name})'
               'CREATE (p)-[:Friend]->(:Person {name: $friend_name})',
               name=name, friend_name=friend_name)


def main():
    # ===== neo4jの設定取得
    config = configparser.ConfigParser()
    config.read('../neo4j.ini')
    uri = config['NEO4J']['uri']
    user = config['NEO4J']['user']
    password = config['NEO4J']['password']

    # neo4jドライバーの作成
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # データベースのクリア
        session.write_transaction(clear_db)

        # ノードの追加
        session.write_transaction(add_person_node, 'taro')

        # FRIEND関係の追加
        session.write_transaction(add_friend_relationship, 'taro', 'hanako')
        # 繰り返しで追加
        friend_list = ['jiro', 'haruka', 'sakura']
        for f in friend_list:
            session.write_transaction(add_friend_relationship, 'taro', f)

        # データの検索
        result = session.read_transaction(search_all)

    # 結果の確認
    for res in result:
        print(res)


if __name__ == '__main__':
    main()