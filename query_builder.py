from typing import Optional


class QueryBuilder:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.columns = "*"
        self.join_tables = []
        self.conditions = []
        self.case_columns = []
        self.group_by = None
        self.order_by = None

    def select(self, *columns: str):
        self.columns = ", ".join(columns)
        return self

    def join(self, table_name: str, join_type: str = 'INNER JOIN', alias: Optional[str] = None,
             on_condition: Optional[str] = None):
        alias = alias or table_name
        self.join_tables.append((table_name, alias, join_type, on_condition))
        return self

    def where(self, condition: str):
        self.conditions.append(condition)
        return self

    def case(self, *case_columns: tuple):
        self.case_columns.extend(case_columns)
        return self

    def group(self, *group_by: str):
        self.group_by = ", ".join(group_by)
        return self

    def order(self, *order_by: str):
        self.order_by = ", ".join(order_by)
        return self

    def build(self) -> str:
        case_statements = []
        for case_column in self.case_columns:
            case_statement = f"CASE WHEN {case_column[0]} THEN {case_column[1]} END"
            case_statements.append(case_statement)

        if case_statements:
            case_string = ", " + ", ".join(case_statements)
        else:
            case_string = ""

        query = f"SELECT {self.columns}{case_string} FROM {self.table_name} AS t1"

        for table_name, alias, join_type, on_condition in self.join_tables:
            query += f" {join_type} {table_name} AS {alias} ON {on_condition}"

        if self.conditions:
            query += " WHERE " + " AND ".join(self.conditions)

        if self.group_by:
            query += f" GROUP BY {self.group_by}"

        if self.order_by:
            query += f" ORDER BY {self.order_by}"

        return query


# Usage
builder1 = QueryBuilder("users")
query = (builder1.select("users.id", "users.name", "departments.department_name")
         .join("departments", "INNER JOIN", "departments", "users.department_id = departments.id")
         .where("age > 18")
         .case(("age < 30", "'Young'"), ("age >= 30", "'Old'"))
         .build())

print(query)
