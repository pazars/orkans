import cx_Oracle

import pandas as pd


def clidata_query(
    query_str,
    lib_dir_path="C:/Program Files/instantclient_21_9",
):
    # Path uz instantclient folderi.
    # Instantclient jāielādē no oracle mājaslapas
    try:
        cx_Oracle.init_oracle_client(lib_dir=lib_dir_path)
    except Exception:
        pass

    try:
        # Izveido savienojumu uz CLIDATA SQL,
        # formas šobrīd ir universālais user lasīšanai,
        # insert/update/delete vajadzētu būt ierobežotam
        connection_cli = cx_Oracle.connect(
            "formas",
            "formas",
            cx_Oracle.makedsn(
                "212.70.174.82",
                "1521",
                service_name="CLIDATA",
            ),
        )
        ### Izveido kursoru ar kuru veic pieprasījumus
        with connection_cli.cursor() as cursor:

            query = query_str

            cursor.execute(query)
            sql_response = cursor.fetchall()
            col_names = [row[0] for row in cursor.description]
            df = pd.DataFrame.from_records(sql_response, columns=col_names)

    finally:
        ### Aizvērt savienojumu pēc darba. Svarīgi!
        if connection_cli is not None:
            connection_cli.close()

    return df


if __name__ == "__main__":

    example_query = """
    select eg_gh_id, ddmmyyyy, value
    from v_day
    where
        eg_gh_id like 'XZV%'
        and eg_el_abbreviation = 'HPRAB'
        and year >= 2021
"""

    print(clidata_query(example_query).head())
