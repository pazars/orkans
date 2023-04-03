import cx_Oracle

import pandas as pd

### Path uz instantclient folderi. Instantclient jāielādē no oracle
cx_Oracle.init_oracle_client(lib_dir="C:/Program Files/instantclient_21_9")

try:
    ### Izveido savienojumu uz CLIDATA SQL, formas šobrīd ir universālais user lasīšanai, insert/update/delete vajadzētu būt ierobežotam
    connection_cli = cx_Oracle.connect(
        "formas",
        "formas",
        cx_Oracle.makedsn("212.70.174.82", "1521", service_name="CLIDATA"),
    )
    ### Izveido kursoru ar kuru veic pieprasījumus
    with connection_cli.cursor() as cursor:

        query = """
        SELECT COUNT(*), COUNT(VALUE)
        FROM v_day
        WHERE
            EG_GH_ID LIKE 'XZV%'
            AND EG_EL_ABBREVIATION = 'HPRAB'
            AND YEAR = 2022
        """

        cursor.execute(query)
        sql_response = cursor.fetchall()
        col_names = [row[0] for row in cursor.description]
        df = pd.DataFrame.from_records(sql_response, columns=col_names)

finally:
    ### Aizvērt savienojumu pēc darba. Svarīgi!
    if connection_cli is not None:
        connection_cli.close()
