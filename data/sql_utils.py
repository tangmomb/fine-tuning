"""Utilitaires SQL partagés par les étapes de fine-tuning et d'évaluation."""


def normalize_sql(sql: str) -> str:
    """Compacte les blancs hors chaînes et identifiants SQL quotés."""
    output = []
    quote = None
    pending_space = False
    index = 0
    while index < len(sql):
        character = sql[index]
        if quote:
            output.append(character)
            if character == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    output.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
        elif character in {"'", '"', chr(96)}:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            output.append(character)
            quote = character
        elif character == ",":
            if output and output[-1] == " ":
                output.pop()
            pending_space = False
            output.append(character)
        elif character.isspace():
            pending_space = True
        else:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            output.append(character)
        index += 1
    return "".join(output).strip()
