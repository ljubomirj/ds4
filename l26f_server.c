#define main l26f_cli_main
#include "test_l26f_multilayer.c"
#undef main

#include "build_release/index.html.h"
#include "build_release/bundle.js.h"
#include "build_release/bundle.css.h"
#include "build_release/loading.html.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#define L26F_SERVER_READ_CAP (16u * 1024u * 1024u)

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} l26f_sbuf;

typedef struct {
    l26f_model model;
    l26f_tokenizer *tok;
    const char *alias;
    pthread_mutex_t lock;
    l26f_session session;
    int session_valid;
} l26f_server_state;

static volatile sig_atomic_t g_l26f_server_stop = 0;
static int g_l26f_server_fd = -1;

static void l26f_server_signal(int sig) {
    (void)sig;
    g_l26f_server_stop = 1;
    if (g_l26f_server_fd >= 0) close(g_l26f_server_fd);
}

static void l26f_sbuf_reserve(l26f_sbuf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) {
        fprintf(stderr, "l26f-server: buffer overflow\n");
        exit(1);
    }
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 1024;
    while (cap < need) cap *= 2;
    char *p = (char *)realloc(b->ptr, cap);
    if (!p) {
        fprintf(stderr, "l26f-server: OOM\n");
        exit(1);
    }
    b->ptr = p;
    b->cap = cap;
}

static void l26f_sbuf_append(l26f_sbuf *b, const void *p, size_t n) {
    l26f_sbuf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void l26f_sbuf_puts(l26f_sbuf *b, const char *s) {
    l26f_sbuf_append(b, s, strlen(s));
}

static void l26f_sbuf_putc(l26f_sbuf *b, char c) {
    l26f_sbuf_append(b, &c, 1);
}

static void l26f_sbuf_printf(l26f_sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    l26f_sbuf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static void l26f_sbuf_free(l26f_sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int l26f_send_all(int fd, const void *p, size_t n) {
    const char *s = (const char *)p;
    while (n > 0) {
        ssize_t w = send(fd, s, n, 0);
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        s += w;
        n -= (size_t)w;
    }
    return 1;
}

static void l26f_json_escape(l26f_sbuf *b, const char *s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  l26f_sbuf_puts(b, "\\\""); break;
        case '\\': l26f_sbuf_puts(b, "\\\\"); break;
        case '\b': l26f_sbuf_puts(b, "\\b"); break;
        case '\f': l26f_sbuf_puts(b, "\\f"); break;
        case '\n': l26f_sbuf_puts(b, "\\n"); break;
        case '\r': l26f_sbuf_puts(b, "\\r"); break;
        case '\t': l26f_sbuf_puts(b, "\\t"); break;
        default:
            if (*p < 0x20) {
                l26f_sbuf_printf(b, "\\u%04x", *p);
            } else {
                l26f_sbuf_putc(b, (char)*p);
            }
            break;
        }
    }
}

static double l26f_json_double(const char *body, const char *key, double def) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    char *end;
    double v = strtod(p, &end);
    if (end == p) return def;
    return v;
}

static const char *l26f_skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static char *l26f_parse_json_string_at(const char **pp) {
    const char *p = l26f_skip_ws(*pp);
    if (*p != '"') return NULL;
    p++;
    l26f_sbuf b = {0};
    while (*p && *p != '"') {
        unsigned char c = (unsigned char)*p++;
        if (c != '\\') {
            l26f_sbuf_putc(&b, (char)c);
            continue;
        }
        c = (unsigned char)*p++;
        switch (c) {
        case '"': l26f_sbuf_putc(&b, '"'); break;
        case '\\': l26f_sbuf_putc(&b, '\\'); break;
        case '/': l26f_sbuf_putc(&b, '/'); break;
        case 'b': l26f_sbuf_putc(&b, '\b'); break;
        case 'f': l26f_sbuf_putc(&b, '\f'); break;
        case 'n': l26f_sbuf_putc(&b, '\n'); break;
        case 'r': l26f_sbuf_putc(&b, '\r'); break;
        case 't': l26f_sbuf_putc(&b, '\t'); break;
        case 'u':
            if (isxdigit((unsigned char)p[0]) && isxdigit((unsigned char)p[1]) &&
                isxdigit((unsigned char)p[2]) && isxdigit((unsigned char)p[3])) {
                unsigned int cp;
                sscanf(p, "%4x", &cp);
                if (cp < 0x80) {
                    l26f_sbuf_putc(&b, (char)cp);
                } else if (cp < 0x800) {
                    l26f_sbuf_putc(&b, (char)(0xC0 | (cp >> 6)));
                    l26f_sbuf_putc(&b, (char)(0x80 | (cp & 0x3F)));
                } else {
                    l26f_sbuf_putc(&b, (char)(0xE0 | (cp >> 12)));
                    l26f_sbuf_putc(&b, (char)(0x80 | ((cp >> 6) & 0x3F)));
                    l26f_sbuf_putc(&b, (char)(0x80 | (cp & 0x3F)));
                }
                p += 4;
            }
            break;
        default:
            l26f_sbuf_putc(&b, (char)c);
            break;
        }
    }
    if (*p == '"') p++;
    *pp = p;
    l26f_sbuf_putc(&b, '\0');
    return b.ptr;
}

static int l26f_json_int(const char *body, const char *key, int def) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    p = l26f_skip_ws(p + 1);
    return atoi(p);
}

static int l26f_json_bool(const char *body, const char *key, int def) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    p = l26f_skip_ws(p + 1);
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

static char *l26f_ascii_strcasestr_local(char *haystack, const char *needle) {
    const size_t nlen = strlen(needle);
    if (nlen == 0) return haystack;
    for (char *p = haystack; *p; p++) {
        size_t i = 0;
        while (i < nlen && p[i] &&
               tolower((unsigned char)p[i]) == tolower((unsigned char)needle[i])) {
            i++;
        }
        if (i == nlen) return p;
    }
    return NULL;
}

static char *l26f_json_string(const char *body, const char *key, const char *def) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(body, pat);
    if (!p) return def ? strdup(def) : NULL;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def ? strdup(def) : NULL;
    p++;
    char *result = l26f_parse_json_string_at(&p);
    return result ? result : (def ? strdup(def) : NULL);
}

typedef struct {
    char *role;
    char *content;
} l26f_chat_msg;

typedef struct {
    l26f_chat_msg *msgs;
    int count;
    int cap;
} l26f_chat_messages;

static void l26f_chat_messages_init(l26f_chat_messages *m) {
    memset(m, 0, sizeof(*m));
}

static void l26f_chat_messages_add(l26f_chat_messages *m, const char *role, char *content) {
    if (m->count >= m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->msgs = (l26f_chat_msg *)realloc(m->msgs, (size_t)m->cap * sizeof(l26f_chat_msg));
    }
    m->msgs[m->count].role = strdup(role);
    m->msgs[m->count].content = content;
    m->count++;
}

static void l26f_chat_messages_free(l26f_chat_messages *m) {
    for (int i = 0; i < m->count; i++) {
        free(m->msgs[i].role);
        free(m->msgs[i].content);
    }
    free(m->msgs);
    memset(m, 0, sizeof(*m));
}

static char *l26f_chat_messages_render(const l26f_chat_messages *m) {
    l26f_sbuf prompt = {0};
    int has_system = 0;
    for (int i = 0; i < m->count; i++) {
        const char *role = m->msgs[i].role;
        const char *content = m->msgs[i].content;
        if (strcmp(role, "system") == 0 && !has_system) {
            l26f_sbuf_puts(&prompt, "<role>SYSTEM</role>");
            l26f_sbuf_puts(&prompt, content);
            l26f_sbuf_puts(&prompt, "\ndetailed thinking off<|role_end|>");
            has_system = 1;
        } else if (strcmp(role, "user") == 0) {
            l26f_sbuf_puts(&prompt, "<role>HUMAN</role>");
            l26f_sbuf_puts(&prompt, content);
            l26f_sbuf_puts(&prompt, "<|role_end|>");
        } else if (strcmp(role, "assistant") == 0) {
            l26f_sbuf_puts(&prompt, "<role>ASSISTANT</role>");
            l26f_sbuf_puts(&prompt, content);
            l26f_sbuf_puts(&prompt, "<|role_end|>");
        }
    }
    if (!has_system) {
        l26f_sbuf_puts(&prompt, "<role>SYSTEM</role>detailed thinking off<|role_end|>");
    }
    l26f_sbuf_puts(&prompt, "<role>ASSISTANT</role>");
    l26f_sbuf_putc(&prompt, '\0');
    return prompt.ptr;
}

static l26f_chat_messages l26f_parse_openai_messages(const char *body) {
    l26f_chat_messages msgs;
    l26f_chat_messages_init(&msgs);
    const char *p = strstr(body, "\"messages\"");
    if (!p) return msgs;
    p = strchr(p, '[');
    if (!p) return msgs;
    p++;
    while (*p) {
        const char *obj_start = strchr(p, '{');
        if (!obj_start) break;
        p = obj_start + 1;
        char *role = NULL;
        char *content = NULL;
        while (*p && *p != '}') {
            if (strncmp(p, "\"role\"", 6) == 0) {
                const char *rp = p + 6;
                rp = strchr(rp, ':');
                if (rp) { rp++; free(role); role = l26f_parse_json_string_at(&rp); }
                p = rp ? rp : p + 1;
            } else if (strncmp(p, "\"content\"", 9) == 0) {
                const char *cp = p + 9;
                cp = strchr(cp, ':');
                if (cp) {
                    cp++;
                    free(content);
                    content = l26f_parse_json_string_at(&cp);
                }
                p = cp ? cp : p + 1;
            } else {
                p++;
            }
        }
        if (content) {
            l26f_chat_messages_add(&msgs, role ? role : "user", content);
        } else {
            free(content);
        }
        free(role);
        if (*p == '}') p++;
    }
    return msgs;
}

static l26f_chat_messages l26f_parse_anthropic_messages(const char *body) {
    l26f_chat_messages msgs;
    l26f_chat_messages_init(&msgs);
    {
        const char *sp = strstr(body, "\"system\"");
        if (sp) {
            sp = strchr(sp + 8, ':');
            if (sp) {
                sp++;
                char *sys = l26f_parse_json_string_at(&sp);
                if (sys) l26f_chat_messages_add(&msgs, "system", sys);
            }
        }
    }
    const char *p = strstr(body, "\"messages\"");
    if (!p) return msgs;
    p = strchr(p, '[');
    if (!p) return msgs;
    p++;
    while (*p) {
        const char *obj_start = strchr(p, '{');
        if (!obj_start) break;
        p = obj_start + 1;
        char *role = NULL;
        char *content = NULL;
        while (*p && *p != '}') {
            if (strncmp(p, "\"role\"", 6) == 0) {
                const char *rp = p + 6;
                rp = strchr(rp, ':');
                if (rp) { rp++; free(role); role = l26f_parse_json_string_at(&rp); }
                p = rp ? rp : p + 1;
            } else if (strncmp(p, "\"content\"", 9) == 0) {
                const char *cp = p + 9;
                cp = strchr(cp, ':');
                if (cp) {
                    cp++;
                    if (*cp == '"') {
                        free(content);
                        content = l26f_parse_json_string_at(&cp);
                        p = cp;
                    } else {
                        const char *arr = strchr(cp, '[');
                        if (arr) {
                            const char *tp = strstr(arr, "\"text\"");
                            if (tp) {
                                tp = strchr(tp + 6, ':');
                                if (tp) {
                                    tp++;
                                    free(content);
                                    content = l26f_parse_json_string_at(&tp);
                                }
                            }
                            p = strchr(arr, ']');
                            if (p) p++;
                        } else {
                            p++;
                        }
                    }
                } else {
                    p++;
                }
            } else {
                p++;
            }
        }
        if (content) {
            l26f_chat_messages_add(&msgs, role ? role : "user", content);
        } else {
            free(content);
        }
        free(role);
        if (*p == '}') p++;
    }
    return msgs;
}

static char *l26f_openai_prompt_from_body(const char *body) {
    l26f_chat_messages msgs = l26f_parse_openai_messages(body);
    if (msgs.count == 0) {
        l26f_chat_messages_free(&msgs);
        return strdup("hello");
    }
    char *result = l26f_chat_messages_render(&msgs);
    l26f_chat_messages_free(&msgs);
    return result;
}

static char *l26f_anthropic_prompt_from_body(const char *body) {
    l26f_chat_messages msgs = l26f_parse_anthropic_messages(body);
    if (msgs.count == 0) {
        l26f_chat_messages_free(&msgs);
        return strdup("hello");
    }
    char *result = l26f_chat_messages_render(&msgs);
    l26f_chat_messages_free(&msgs);
    return result;
}

static void l26f_http_response(int fd, int status, const char *status_text,
        const char *content_type, const char *extra_headers,
        const char *body_data, size_t body_len) {
    l26f_sbuf h = {0};
    l26f_sbuf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n",
        status, status_text, content_type, body_len);
    if (extra_headers) l26f_sbuf_puts(&h, extra_headers);
    l26f_sbuf_puts(&h, "\r\n");
    l26f_send_all(fd, h.ptr, h.len);
    l26f_sbuf_free(&h);
    if (body_data && body_len) l26f_send_all(fd, body_data, body_len);
}

static void l26f_http_json(int fd, int status, const char *status_text, const char *json) {
    l26f_http_response(fd, status, status_text, "application/json", NULL,
        json, strlen(json));
}

static void l26f_http_error(int fd, int status, const char *error_type, const char *message) {
    const char *st;
    switch (status) {
        case 400: st = "Bad Request"; break;
        case 404: st = "Not Found"; break;
        case 500: st = "Internal Server Error"; break;
        default: st = "Error"; break;
    }
    l26f_sbuf json = {0};
    l26f_sbuf_puts(&json, "{\"error\":{\"type\":\"");
    l26f_json_escape(&json, error_type);
    l26f_sbuf_puts(&json, "\",\"message\":\"");
    l26f_json_escape(&json, message);
    l26f_sbuf_puts(&json, "\"}}");
    l26f_http_json(fd, status, st, json.ptr);
    l26f_sbuf_free(&json);
}

static int l26f_sse_begin(int fd) {
    const char *h =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Connection: close\r\n\r\n";
    return l26f_send_all(fd, h, strlen(h));
}

static int l26f_sse_send(int fd, const char *event, const char *data) {
    l26f_sbuf b = {0};
    if (event) l26f_sbuf_printf(&b, "event: %s\n", event);
    l26f_sbuf_printf(&b, "data: %s\n\n", data);
    int ok = l26f_send_all(fd, b.ptr, b.len);
    l26f_sbuf_free(&b);
    return ok;
}

static int l26f_sse_openai_token(int fd, const char *alias, int32_t token_id,
        const char *piece, const char *model_id) {
    l26f_sbuf b = {0};
    l26f_sbuf_puts(&b, "{\"id\":\"");
    l26f_sbuf_puts(&b, model_id ? model_id : "chatcmpl-l26f");
    l26f_sbuf_puts(&b, "\",\"object\":\"chat.completion.chunk\",\"model\":\"");
    l26f_json_escape(&b, alias);
    l26f_sbuf_puts(&b, "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"");
    l26f_json_escape(&b, piece);
    l26f_sbuf_printf(&b, "\"},\"token_ids\":[%d],\"finish_reason\":null}]}", token_id);
    int ok = l26f_sse_send(fd, NULL, b.ptr);
    l26f_sbuf_free(&b);
    return ok;
}

static int l26f_sse_openai_done(int fd, const char *alias, int prompt_tokens,
        int completion_tokens, const char *model_id) {
    l26f_sbuf b = {0};
    l26f_sbuf_puts(&b, "{\"id\":\"");
    l26f_sbuf_puts(&b, model_id ? model_id : "chatcmpl-l26f");
    l26f_sbuf_puts(&b, "\",\"object\":\"chat.completion.chunk\",\"model\":\"");
    l26f_json_escape(&b, alias);
    l26f_sbuf_puts(&b, "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
    l26f_sse_send(fd, NULL, b.ptr);
    l26f_sbuf_free(&b);

    l26f_sbuf b2 = {0};
    l26f_sbuf_puts(&b2, "{\"id\":\"");
    l26f_sbuf_puts(&b2, model_id ? model_id : "chatcmpl-l26f");
    l26f_sbuf_puts(&b2, "\",\"object\":\"chat.completion.chunk\",\"model\":\"");
    l26f_json_escape(&b2, alias);
    l26f_sbuf_printf(&b2, "\",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},\"choices\":[]}",
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    l26f_sse_send(fd, NULL, b2.ptr);
    l26f_sbuf_free(&b2);
    return l26f_sse_send(fd, NULL, "[DONE]");
}

static int l26f_sse_anthropic_start(int fd, const char *alias, const char *msg_id) {
    l26f_sbuf b = {0};
    l26f_sbuf_puts(&b, "{\"type\":\"message_start\",\"message\":{\"id\":\"");
    l26f_sbuf_puts(&b, msg_id);
    l26f_sbuf_puts(&b, "\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"");
    l26f_json_escape(&b, alias);
    l26f_sbuf_puts(&b, "\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}");
    int ok = l26f_sse_send(fd, "message_start", b.ptr);
    l26f_sbuf_free(&b);
    return ok;
}

static int l26f_sse_anthropic_text_start(int fd) {
    return l26f_sse_send(fd, "content_block_start",
        "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
}

static int l26f_sse_anthropic_text_delta(int fd, const char *piece) {
    l26f_sbuf b = {0};
    l26f_sbuf_puts(&b, "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"");
    l26f_json_escape(&b, piece);
    l26f_sbuf_puts(&b, "\"}}");
    int ok = l26f_sse_send(fd, "content_block_delta", b.ptr);
    l26f_sbuf_free(&b);
    return ok;
}

static int l26f_sse_anthropic_text_stop(int fd) {
    return l26f_sse_send(fd, "content_block_stop",
        "{\"type\":\"content_block_stop\",\"index\":0}");
}

static int l26f_sse_anthropic_done(int fd, int input_tokens, int output_tokens,
        const char *stop_reason) {
    (void)input_tokens;
    l26f_sbuf b = {0};
    l26f_sbuf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"");
    l26f_sbuf_puts(&b, stop_reason);
    l26f_sbuf_puts(&b, "\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    l26f_sbuf_printf(&b, "%d}}", output_tokens);
    l26f_sse_send(fd, "message_delta", b.ptr);
    l26f_sbuf_free(&b);
    return l26f_sse_send(fd, "message_stop", "{\"type\":\"message_stop\"}");
}

static int l26f_server_prefill(l26f_server_state *st, l26f_session *session,
        const int32_t *tokens, int n_tokens) {
    if (n_tokens <= 0) return 0;
    for (int i = 0; i < n_tokens; i++) {
        if (!l26f_embed_token(session, (uint32_t)tokens[i])) return 0;
        if (!l26f_forward_pass(session, &st->model, i, false)) return 0;
    }
    return 1;
}

static int l26f_server_sample_token(l26f_session *session, l26f_tokenizer *tok,
        float temperature, int top_k, float top_p, uint64_t *rng) {
    (void)tok;
    if (temperature <= 0.0f) {
        return l26f_output_greedy_token(session);
    }
    if (!l26f_output_logits(session)) return -1;
    l26f_sample_params params = {
        .temperature = temperature,
        .top_k = top_k > 0 ? top_k : 40,
        .top_p = top_p > 0.0f && top_p < 1.0f ? top_p : 0.95f,
        .seed = 0
    };
    return l26f_sample(session, &params, rng);
}

typedef enum {
    L26F_API_OPENAI_CHAT,
    L26F_API_OPENAI_COMPLETIONS,
    L26F_API_ANTHROPIC_MESSAGES
} l26f_api_type;

static int l26f_server_generate(l26f_server_state *st, int fd,
        const char *body, l26f_api_type api) {
    int max_tokens_req = l26f_json_int(body, "max_completion_tokens", -1);
    if (max_tokens_req <= 0) max_tokens_req = l26f_json_int(body, "max_tokens", -1);
    const int max_tokens = max_tokens_req > 0 ? max_tokens_req : 256;
    const int stream = l26f_json_bool(body, "stream", 0);
    const double temperature = l26f_json_double(body, "temperature", 0.0);
    const int top_k = l26f_json_int(body, "top_k", 40);
    const double top_p = l26f_json_double(body, "top_p", 0.95);

    char *prompt = NULL;
    if (api == L26F_API_ANTHROPIC_MESSAGES) {
        prompt = l26f_anthropic_prompt_from_body(body);
    } else if (api == L26F_API_OPENAI_COMPLETIONS) {
        char *pp = l26f_json_string(body, "prompt", "hello");
        if (pp) { prompt = pp; } else { prompt = strdup("hello"); }
    } else {
        prompt = l26f_openai_prompt_from_body(body);
    }

    fprintf(stderr, "l26f-server: request api=%d prompt_bytes=%zu max_tokens=%d stream=%d temp=%.2f\n",
            api, strlen(prompt), max_tokens, stream, temperature);

    pthread_mutex_lock(&st->lock);

    int max_prompt_tokens = (int)strlen(prompt) * 2 + 64;
    if (max_prompt_tokens < 64) max_prompt_tokens = 64;
    if (max_prompt_tokens > 8192) max_prompt_tokens = 8192;
    int32_t *prompt_tokens = (int32_t *)malloc((uint64_t)max_prompt_tokens * sizeof(int32_t));
    if (!prompt_tokens) {
        pthread_mutex_unlock(&st->lock);
        free(prompt);
        l26f_http_error(fd, 500, "internal_error", "prompt token allocation failed");
        return 0;
    }
    int n_prompt = l26f_text_encode(st->tok, prompt, prompt_tokens, max_prompt_tokens);
    if (n_prompt <= 0) {
        prompt_tokens[0] = 1;
        n_prompt = 1;
    }
    fprintf(stderr, "l26f-server: encoded prompt_tokens=%d\n", n_prompt);

    l26f_session session;
    int ok = l26f_session_init(&session, &st->model);
    if (!ok) {
        pthread_mutex_unlock(&st->lock);
        free(prompt_tokens);
        free(prompt);
        l26f_http_error(fd, 500, "internal_error", "session init failed");
        return 0;
    }

    ok = l26f_server_prefill(st, &session, prompt_tokens, n_prompt);
    if (!ok) {
        l26f_session_free(&session);
        pthread_mutex_unlock(&st->lock);
        free(prompt_tokens);
        free(prompt);
        l26f_http_error(fd, 500, "internal_error", "prefill failed");
        return 0;
    }

    if (stream && (api == L26F_API_OPENAI_CHAT || api == L26F_API_OPENAI_COMPLETIONS)) {
        l26f_sse_begin(fd);
    } else if (stream && api == L26F_API_ANTHROPIC_MESSAGES) {
        l26f_sse_begin(fd);
        l26f_sse_anthropic_start(fd, st->alias, "msg_l26f");
        l26f_sse_anthropic_text_start(fd);
    }

    l26f_sbuf full = {0};
    int completion_tokens = 0;
    int32_t current_token = -1;
    uint64_t rng = (uint64_t)time(NULL);
    float temp = (float)temperature;
    int tk = top_k;
    float tp = (float)top_p;

    for (int i = 0; i < max_tokens; i++) {
        int32_t next_token = l26f_server_sample_token(&session, st->tok, temp, tk, tp, &rng);
        if (next_token < 0) break;
        char piece[256];
        l26f_token_decode(st->tok, next_token, piece, sizeof(piece));
        l26f_sbuf_puts(&full, piece);
        completion_tokens++;

        if (stream) {
            if (api == L26F_API_OPENAI_CHAT) {
                if (!l26f_sse_openai_token(fd, st->alias, next_token, piece, "chatcmpl-l26f")) break;
            } else if (api == L26F_API_OPENAI_COMPLETIONS) {
                if (!l26f_sse_openai_token(fd, st->alias, next_token, piece, "cmpl-l26f")) break;
            } else if (api == L26F_API_ANTHROPIC_MESSAGES) {
                if (!l26f_sse_anthropic_text_delta(fd, piece)) break;
            }
        }

        current_token = next_token;
        if (next_token == st->tok->eos_id) break;
        if (i + 1 < max_tokens) {
            if (!l26f_embed_token(&session, (uint32_t)current_token)) break;
            if (!l26f_forward_pass(&session, &st->model, n_prompt + i, false)) break;
        }
    }

    if (stream) {
        if (api == L26F_API_OPENAI_CHAT) {
            l26f_sse_openai_done(fd, st->alias, n_prompt, completion_tokens, "chatcmpl-l26f");
        } else if (api == L26F_API_OPENAI_COMPLETIONS) {
            l26f_sse_openai_done(fd, st->alias, n_prompt, completion_tokens, "cmpl-l26f");
        } else if (api == L26F_API_ANTHROPIC_MESSAGES) {
            l26f_sse_anthropic_text_stop(fd);
            l26f_sse_anthropic_done(fd, n_prompt, completion_tokens,
                current_token == st->tok->eos_id ? "end_turn" : "max_tokens");
        }
    } else {
        const char *finish_reason = current_token == st->tok->eos_id ? "stop" : "length";
        if (api == L26F_API_OPENAI_CHAT) {
            l26f_sbuf json = {0};
            l26f_sbuf_puts(&json, "{\"id\":\"chatcmpl-l26f\",\"object\":\"chat.completion\",\"model\":\"");
            l26f_json_escape(&json, st->alias);
            l26f_sbuf_puts(&json, "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"");
            l26f_json_escape(&json, full.ptr ? full.ptr : "");
            l26f_sbuf_printf(&json, "\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                    finish_reason, n_prompt, completion_tokens, n_prompt + completion_tokens);
            l26f_http_json(fd, 200, "OK", json.ptr);
            l26f_sbuf_free(&json);
        } else if (api == L26F_API_OPENAI_COMPLETIONS) {
            l26f_sbuf json = {0};
            l26f_sbuf_puts(&json, "{\"id\":\"cmpl-l26f\",\"object\":\"text_completion\",\"model\":\"");
            l26f_json_escape(&json, st->alias);
            l26f_sbuf_puts(&json, "\",\"choices\":[{\"index\":0,\"text\":\"");
            l26f_json_escape(&json, full.ptr ? full.ptr : "");
            l26f_sbuf_printf(&json, "\",\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                    finish_reason, n_prompt, completion_tokens, n_prompt + completion_tokens);
            l26f_http_json(fd, 200, "OK", json.ptr);
            l26f_sbuf_free(&json);
        } else if (api == L26F_API_ANTHROPIC_MESSAGES) {
            l26f_sbuf json = {0};
            l26f_sbuf_puts(&json, "{\"id\":\"msg_l26f\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"");
            l26f_json_escape(&json, st->alias);
            l26f_sbuf_puts(&json, "\",\"content\":[{\"type\":\"text\",\"text\":\"");
            l26f_json_escape(&json, full.ptr ? full.ptr : "");
            l26f_sbuf_printf(&json, "\"}],\"stop_reason\":\"%s\",\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d}}",
                    current_token == st->tok->eos_id ? "end_turn" : "max_tokens",
                    n_prompt, completion_tokens);
            l26f_http_json(fd, 200, "OK", json.ptr);
            l26f_sbuf_free(&json);
        }
    }

    l26f_sbuf_free(&full);
    l26f_session_free(&session);
    pthread_mutex_unlock(&st->lock);
    free(prompt_tokens);
    free(prompt);
    fprintf(stderr, "l26f-server: complete completion_tokens=%d\n", completion_tokens);
    return 1;
}

static int l26f_read_request(int fd, char **out) {
    l26f_sbuf b = {0};
    char tmp[8192];
    size_t header_end = 0;
    size_t content_length = 0;
    while (b.len < L26F_SERVER_READ_CAP) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;
        l26f_sbuf_append(&b, tmp, (size_t)n);
        char *he = strstr(b.ptr, "\r\n\r\n");
        if (he) {
            header_end = (size_t)(he - b.ptr) + 4;
            char *cl = l26f_ascii_strcasestr_local(b.ptr, "Content-Length:");
            if (cl && (size_t)(cl - b.ptr) < header_end) {
                content_length = (size_t)strtoull(cl + strlen("Content-Length:"), NULL, 10);
            }
            if (b.len >= header_end + content_length) break;
        }
    }
    if (!b.ptr) return 0;
    *out = b.ptr;
    return 1;
}

static void l26f_handle_client(int fd, l26f_server_state *st) {
    char *req = NULL;
    if (!l26f_read_request(fd, &req)) {
        close(fd);
        return;
    }
    char method[16] = {0};
    char path[256] = {0};
    sscanf(req, "%15s %255s", method, path);
    char *qs = strchr(path, '?');
    if (qs) *qs = '\0';
    fprintf(stderr, "  %s %s\n", method, path);
    char *body = strstr(req, "\r\n\r\n");
    body = body ? body + 4 : req + strlen(req);

    if (strcmp(method, "OPTIONS") == 0) {
        l26f_http_response(fd, 204, "No Content", "text/plain",
            "Connection: close\r\n", NULL, 0);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        l26f_http_response(fd, 200, "OK", "text/html; charset=utf-8", NULL,
            (const char *)l26f_webui_index_html, (size_t)l26f_webui_index_html_len);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/bundle.js") == 0) {
        l26f_http_response(fd, 200, "OK", "application/javascript; charset=utf-8", NULL,
            (const char *)l26f_webui_bundle_js, (size_t)l26f_webui_bundle_js_len);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/bundle.css") == 0) {
        l26f_http_response(fd, 200, "OK", "text/css; charset=utf-8", NULL,
            (const char *)l26f_webui_bundle_css, (size_t)l26f_webui_bundle_css_len);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/loading.html") == 0) {
        l26f_http_response(fd, 200, "OK", "text/html; charset=utf-8", NULL,
            (const char *)l26f_webui_loading_html, (size_t)l26f_webui_loading_html_len);
    } else if ((strcmp(method, "GET") == 0 || strcmp(method, "POST") == 0) &&
            strcmp(path, "/props") == 0) {
        l26f_sbuf json = {0};
        l26f_sbuf_puts(&json,
            "{\"default_generation_settings\":{"
            "\"n_predict\":-1,\"temperature\":0.0,\"top_k\":40,\"top_p\":0.95,"
            "\"n_keep\":0,\"stream\":true"
            "},"
            "\"total_slots\":1,"
            "\"model_alias\":\"");
        l26f_json_escape(&json, st->alias);
        l26f_sbuf_puts(&json,
            "\","
            "\"modalities\":{\"vision\":false,\"audio\":false},"
            "\"endpoint_slots\":false,"
            "\"endpoint_props\":false,"
            "\"endpoint_metrics\":false,"
            "\"webui\":true,"
            "\"is_sleeping\":false"
            "}");
        l26f_http_json(fd, 200, "OK", json.ptr);
        l26f_sbuf_free(&json);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        l26f_http_json(fd, 200, "OK", "{\"status\":\"ok\"}");
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/tools") == 0) {
        l26f_http_json(fd, 200, "OK", "[]");
    } else if (strcmp(method, "HEAD") == 0 && strcmp(path, "/cors-proxy") == 0) {
        l26f_http_response(fd, 404, "Not Found", "text/plain", NULL, NULL, 0);
    } else if (strcmp(method, "GET") == 0 &&
            (strcmp(path, "/v1/models") == 0 || strcmp(path, "/models") == 0)) {
        l26f_sbuf json = {0};
        l26f_sbuf_puts(&json, "{\"object\":\"list\",\"data\":[{\"id\":\"");
        l26f_json_escape(&json, st->alias);
        l26f_sbuf_puts(&json, "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"l26f\"}]}");
        l26f_http_json(fd, 200, "OK", json.ptr);
        l26f_sbuf_free(&json);
    } else if (strcmp(method, "POST") == 0 &&
            (strcmp(path, "/v1/chat/completions") == 0 || strcmp(path, "/chat/completions") == 0)) {
        l26f_server_generate(st, fd, body, L26F_API_OPENAI_CHAT);
    } else if (strcmp(method, "POST") == 0 &&
            (strcmp(path, "/v1/completions") == 0 || strcmp(path, "/completions") == 0)) {
        l26f_server_generate(st, fd, body, L26F_API_OPENAI_COMPLETIONS);
    } else if (strcmp(method, "POST") == 0 &&
            (strcmp(path, "/v1/messages") == 0 || strcmp(path, "/messages") == 0)) {
        l26f_server_generate(st, fd, body, L26F_API_ANTHROPIC_MESSAGES);
    } else {
        l26f_http_error(fd, 404, "not_found", "not found");
    }
    free(req);
    close(fd);
}

typedef struct {
    int fd;
    l26f_server_state *st;
} l26f_client_job;

static void *l26f_client_thread(void *arg) {
    l26f_client_job *job = (l26f_client_job *)arg;
    l26f_handle_client(job->fd, job->st);
    free(job);
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *host = "127.0.0.1";
    const char *alias = "inclusionAI/Ling-2.6-flash";
    int port = 8081;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--alias") == 0 && i + 1 < argc) {
            alias = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "l26f-server: OpenAI/Anthropic-compatible inference server\n\n"
                "usage: %s --model MODEL.gguf [options]\n\n"
                "options:\n"
                "  -m, --model PATH    GGUF model path (required)\n"
                "  --host ADDR         bind address (default 127.0.0.1)\n"
                "  --port PORT         bind port (default 8081)\n"
                "  --alias NAME        model name (default inclusionAI/Ling-2.6-flash)\n"
                "  -h, --help          show this help\n\n"
                "endpoints:\n"
                "  GET  /v1/models              list models\n"
                "  POST /v1/chat/completions    OpenAI chat completions\n"
                "  POST /v1/completions         OpenAI text completions\n"
                "  POST /v1/messages            Anthropic messages API\n"
                "  OPTIONS *                     CORS preflight\n\n"
                "env vars:\n"
                "  L26F_FUSED_MOE=1     enable fused MoE (recommended)\n"
                "  L26F_LOGITS_FUSE=0   disable fused logits+argmax\n"
                "  L26F_GLA_QKV_GATE_FUSE=0  disable GLA QKV+gate fusion\n",
                argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s (use --help)\n", argv[i]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "l26f-server: --model is required (use --help)\n");
        return 1;
    }

    l26f_server_state st = {0};
    st.alias = alias;
    pthread_mutex_init(&st.lock, NULL);

    fprintf(stderr, "l26f-server: loading model %s\n", model_path);
    l26f_model_open(&st.model, model_path);
    st.tok = l26f_tokenizer_from_model(&st.model);
    if (!st.tok) {
        fprintf(stderr, "l26f-server: tokenizer load failed\n");
        return 1;
    }
    if (!ds4_metal_init()) {
        fprintf(stderr, "l26f-server: Metal init failed\n");
        return 1;
    }
    uint64_t model_tensor_off = 0;
    uint64_t model_tensor_bytes = st.model.size;
    if (!XMODEL_TENSOR_RANGE(&st.model, &model_tensor_off, &model_tensor_bytes)) return 1;
    if (!ds4_metal_set_model_map_range(st.model.map, st.model.size,
            model_tensor_off, model_tensor_bytes)) return 1;

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "l26f-server: invalid IPv4 host %s\n", host);
        return 1;
    }
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(listen_fd, 16) < 0) {
        perror("listen");
        return 1;
    }

    signal(SIGINT, l26f_server_signal);
    signal(SIGTERM, l26f_server_signal);
    signal(SIGPIPE, SIG_IGN);
    g_l26f_server_fd = listen_fd;
    fprintf(stderr, "l26f-server: listening on http://%s:%d/v1\n", host, port);
    fprintf(stderr, "l26f-server: endpoints: /v1/models /v1/chat/completions /v1/completions /v1/messages\n");

    while (!g_l26f_server_stop) {
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (g_l26f_server_stop) break;
            perror("accept");
            continue;
        }
        l26f_client_job *job = (l26f_client_job *)malloc(sizeof(*job));
        if (!job) {
            close(fd);
            continue;
        }
        job->fd = fd;
        job->st = &st;
        pthread_t th;
        if (pthread_create(&th, NULL, l26f_client_thread, job) == 0) {
            pthread_detach(th);
        } else {
            l26f_handle_client(fd, &st);
            free(job);
        }
    }

    fprintf(stderr, "l26f-server: shutting down\n");
    ds4_metal_cleanup();
    l26f_model_close(&st.model);
    l26f_tokenizer_close(st.tok);
    pthread_mutex_destroy(&st.lock);
    close(listen_fd);
    return 0;
}
