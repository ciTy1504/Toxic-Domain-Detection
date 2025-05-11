def check_content(text):
    if not text or all(line.strip() == "" or line.strip() == "-->" for line in text):
        return "blank"
    joined_text = " ".join(text).replace("\n", " ")

    if len(joined_text) < 20:
        return "short"

    if any(x in joined_text for x in [
        "This site can’t be reached",
        "Page cannot be displayed",
        "This site is temporarily unavailable"
    ]) and len(joined_text) < 100:
        return "can_reach"

    if (("Warning" in joined_text and "Suspected Phishing" in joined_text
        and "This website has been reported for potential phishing." in joined_text)
        or "You've ended up on one of our phishing domains" in joined_text
        or "You are seeing this warning because this site does not support HTTPS" in joined_text
        or "doesn’t support a secure connection" in joined_text
    ) and len(joined_text) < 100:
        return "warn"

    if "Your connection is not private" in joined_text and len(joined_text) < 100:
        return "private"

    if any(e in joined_text for e in [
        "400 Bad Request", "401 Unauthorized", "402 Payment Required", "403 Forbidden",
        "404", "404 Not Found", "405 Method Not Allowed", "406 Not Acceptable",
        "407 Proxy Authentication Required", "408 Request Timeout", "409 Conflict",
        "410 Gone", "411 Length Required", "412 Precondition Failed", "413 Payload Too Large",
        "414 URI Too Long", "415 Unsupported Media Type", "416 Range Not Satisfiable",
        "417 Expectation Failed", "418 I'm a teapot",
        "422 Unprocessable Entity", "429 Too Many Requests",
        "500", "500 Internal Server Error", "501 Not Implemented", "502 Bad Gateway",
        "503 Service Unavailable", "504 Gateway Timeout", "505 HTTP Version Not Supported",
        "HTTP ERROR 500", "HTTP ERROR 520", "HTTP ERROR 522", "HTTP ERROR 443", "HTTP ERROR",
        "Error 522", "Error 520", "Error 525 SSL handshake failed", "Error 523 Origin is unreachable",
        "Page not found", "This page isn’t working", "The page isn’t redirecting properly",
        "An error occurred while processing your request", "Temporarily unavailable", "ERR_SSL_PROTOCOL_ERROR",
        "Something went wrong", "Service unavailable", "Server error", "Website is down",
        "Connection timed out", "Refused to connect", "Unable to load the page"
    ]) and len(joined_text) < 100:
        return "error"

    return ""

