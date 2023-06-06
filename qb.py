from typing import Optional


class QueryBuilder:
    """A class to build SQL queries in a Pythonic way."""

    def __init__(self, table_name: str = None, subquery: str = None, alias: str = 't1'):
        """Initialize a QueryBuilder instance."""
        if not table_name and not subquery:
            raise ValueError("Either table_name or subquery should be provided")

        self.table_name = table_name
        self.subquery = subquery
        self.alias = alias
        self.columns = "*"
        self.join_tables = []
        self.conditions = {}
        self.case_columns = []
        self.group_by = None
        self.order_by = None

    def select(self, *columns: str):
        """Specify the columns to select."""
        if not columns:
            raise ValueError("At least one column must be provided")
        self.columns = ", ".join(columns)
        return self

    def join(self, table_name: str, join_type: str, alias: str, on: str):
        """Join another table."""
        if not table_name or not join_type or not alias or not on:
            raise ValueError("All join parameters must be provided")
        self.join_tables.append((table_name, join_type, alias, on))
        return self

    def where(self, **conditions):
        """Add conditions to the WHERE clause."""
        if not conditions:
            raise ValueError("At least one condition must be provided")
        self.conditions.update(conditions)
        return self

    def case(self, case_column: str, *conditions: tuple, else_result=None):
        """Add a CASE statement."""
        if not case_column or not conditions:
            raise ValueError("case_column and at least one condition must be provided")
        self.case_columns.append((case_column, conditions, else_result))
        return self

    def group(self, *group_by: str):
        """Add a GROUP BY clause."""
        if not group_by:
            raise ValueError("At least one group_by column must be provided")
        self.group_by = ", ".join(group_by)
        return self

    def order(self, *order_by: str):
        """Add an ORDER BY clause."""
        if not order_by:
            raise ValueError("At least one order_by column must be provided")
        self.order_by = ", ".join(order_by)
        return self

    def build(self) -> str:
        """Build the SQL query."""
        if not self.table_name and not self.subquery:
            raise ValueError("Either table_name or subquery must be provided")

        from_clause = f"{self.table_name} AS {self.alias}" if self.table_name else f"({self.subquery}) AS {self.alias}"
        join_clause = " ".join(
            [f"{join_type} JOIN {table_name} AS {alias} ON {on}" for table_name, join_type, alias, on in
             self.join_tables])
        where_clause = " AND ".join([f"{k}={v}" for k, v in self.conditions.items()]) if self.conditions else None
        group_by_clause = f"GROUP BY {self.group_by}" if self.group_by else ""
        order_by_clause = f"ORDER BY {self.order_by}" if self.order_by else ""

        case_statements = []
        for case_column, conditions, else_result in self.case_columns:
            case_statement = f"CASE {case_column} " + " ".join(
                [f"WHEN {condition[0]} THEN {condition[1]}" for condition in conditions])
            if else_result:
                case_statement += f" ELSE {else_result}"
            case_statements.append(case_statement + " END")
        case_string = ", " + ", ".join(case_statements) if case_statements else ""

        query = f"SELECT {self.columns}{case_string} FROM {from_clause} {join_clause}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f" {group_by_clause} {order_by_clause}"

        return query.strip()

    def create_table_from_select(self, table_name: str) -> str:
        """Create a new table from a SELECT query."""
        if not self.subquery:
            raise ValueError("Subquery is not defined")
        query = f"""
        CREATE TABLE {table_name} AS
        ({self.subquery})
        WITH DATA;
        """
        return query

    def create_volatile_table_from_select(self, table_name: str, on_commit: str = 'PRESERVE ROWS') -> str:
        """Create a new volatile table from a SELECT query."""
        if not self.subquery:
            raise ValueError("Subquery is not defined")

        if on_commit not in ['PRESERVE ROWS', 'DELETE ROWS', 'DROP TABLE']:
            raise ValueError(
                "Invalid on_commit value. It must be one of 'PRESERVE ROWS', 'DELETE ROWS', or 'DROP TABLE'")

        query = f"""
        CREATE VOLATILE TABLE {table_name} AS
        ({self.subquery})
        WITH DATA
        ON COMMIT {on_commit};
        """
        return query

    def create_cte(self, cte_name: str) -> str:
        """Create a Common Table Expression (CTE)."""

        query = f"""
        WITH {cte_name} AS (
        {self.subquery}
        )
        """
        return query

    # Initialize the main table
main_query = QueryBuilder(table_name="users")
main_query = (main_query.select("id", "name", "age", "country")
              .join(table_name="orders", join_type="LEFT", alias="o", on="users.id = o.user_id")
              .where(age__gt=18)
              .case("age",
                    ("age < 30", "'Young'"),
                    ("age >= 30", "'Old'"),
                    else_result="'Unknown'")
              .group("users.id", "users.name", "users.age", "users.country")
              .order("users.age DESC", "users.name ASC"))

# CTE creation
cte_query = main_query.create_cte(cte_name="user_orders")

# Define a subquery using the CTE
subquery = f"SELECT * FROM user_orders WHERE country = 'USA'"

# Initialize a new QueryBuilder with the subquery
usa_query = QueryBuilder(subquery=subquery, alias="usa_users")
usa_query = usa_query.select("id", "name", "age", "country")

# Build the final query
final_query = usa_query.build()
print(final_query)

# Create a volatile table from the subquery
volatile_table_query = usa_query.create_volatile_table_from_select(table_name="usa_user_orders",
                                                                   on_commit="PRESERVE ROWS")
print(volatile_table_query)
print(main_query.build())
