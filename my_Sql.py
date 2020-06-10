"""
Install : sudo apt install mysql-server
Installing mysql clinet which is fork of mySQLdb
sudo apt install python3-dev
sudo apt install python3-dev libmysqlclient-dev default-libmysqlclient-dev
pip install mysqlclient
usage: sudo mysql -u root
show databases;
create database database_name;
CREATE USER 'teleuser'@'%' IDENTIFIED WITH mysql_native_password BY 'password';
GRANT ALL ON databasename.* TO 'teleuser'@'%';
flush privileges; # to reflect the changes we made

"""
import pymysql

try:
    # Connect to the database
    connection = pymysql.connect(host='localhost',
                             user='telegram_user',
                             password='tele123',
                             db='telegram')


    cursor=connection.cursor()
    #cur.execute("SELECT Host,User FROM user")

    #print(cur.description)

    #print()

    #for row in cur:
    #    print(row)

    #cur.close()

    # Create a new record
    stmt = "SHOW TABLES LIKE 'telegram'"
    cursor.execute(stmt)
    result = cursor.fetchone()
    if not result:
        # there is a table named "tableName"
        sql = "create table telegram (EmployeeID int, Ename varchar(20), DeptID int, Salary int, Dname varchar(20), Dlocation varchar(20))"
        cursor.execute(sql)
    sql1 ="INSERT INTO `telegram` (`EmployeeID`, `Ename`, `DeptID`, `Salary`, `Dname`, `Dlocation`) VALUES (%s, %s, %s, %s, %s, %s)"
    cursor.execute(sql1, (1009,'Morgan',1,4000,'HR','Mumbai'))

    # connection is not autocommit by default. So we must commit to save our changes.
    connection.commit()

    # Execute query
    sql = "SELECT * FROM `telegram`"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        print(i)

except Exception as e:
    print(e)

finally:
    # close the database connection using close() method.
    connection.close()
