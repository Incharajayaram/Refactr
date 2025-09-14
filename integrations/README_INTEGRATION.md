# Git Integration Guide

This guide explains how to set up GitHub and GitLab integrations for the Code Quality Intelligence Agent to automatically analyze pull requests and merge requests.

## 🚀 Overview

The integration allows the Code Quality Agent to:
- Receive webhook events when PRs/MRs are opened or updated
- Automatically analyze the code in the PR/MR
- Post analysis results as comments with actionable insights
- Track and update existing comments to avoid spam

## 🔧 GitHub Integration

### Prerequisites

1. **GitHub Personal Access Token**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate a new token with these scopes:
     - `repo` (full access) OR more restrictive:
       - `public_repo` (for public repositories only)
       - `repo:status`
       - `repo_deployment`
       - `pull_request` (read/write access to pull requests)
   - Save the token securely

2. **Webhook Secret**
   - Generate a secure random string: `openssl rand -hex 32`
   - Keep this secret for webhook configuration

### Server Configuration

Set these environment variables on your server:

```bash
export GITHUB_TOKEN="ghp_your_personal_access_token"
export GITHUB_WEBHOOK_SECRET="your_webhook_secret"
```

### Webhook Setup

1. Go to your repository → Settings → Webhooks → Add webhook

2. Configure the webhook:
   - **Payload URL**: `https://your-server.com/webhook/github`
   - **Content type**: `application/json`
   - **Secret**: Enter the same value as `GITHUB_WEBHOOK_SECRET`
   - **SSL verification**: Enable (recommended)
   - **Which events?**: Select "Let me select individual events"
     - Check only: **Pull requests**
   - **Active**: ✓ Check this box

3. Click "Add webhook"

### Testing

Test with a sample payload:
```bash
python -m integrations.github_integration simulate \
  --payload samples/github_pr_opened.json \
  --skip-post
```

### Troubleshooting

- **Signature verification failed**: Ensure `GITHUB_WEBHOOK_SECRET` matches exactly
- **401 Unauthorized**: Check your `GITHUB_TOKEN` is valid and has correct scopes
- **Rate limiting**: The integration waits 5 minutes between comments on the same repository

## 🦊 GitLab Integration

### Prerequisites

1. **GitLab Personal Access Token**
   - Go to GitLab → User Settings → Access Tokens
   - Create a token with these scopes:
     - `api` (full API access) OR more restrictive:
       - `read_api`
       - `read_repository`
       - `write_repository`
   - Save the token securely

2. **Webhook Token** (optional but recommended)
   - Generate a secure token: `openssl rand -hex 32`

### Server Configuration

```bash
export GITLAB_TOKEN="glpat_your_personal_access_token"
export GITLAB_WEBHOOK_TOKEN="your_webhook_token"  # Optional
export GITLAB_URL="https://gitlab.com"  # Or your GitLab instance URL
```

### Webhook Setup

1. Go to your project → Settings → Webhooks

2. Configure the webhook:
   - **URL**: `https://your-server.com/webhook/gitlab`
   - **Secret token**: Enter the same value as `GITLAB_WEBHOOK_TOKEN`
   - **Trigger**: Select only "Merge request events"
   - **SSL verification**: Enable
   - **Enable**: ✓ Check this box

3. Click "Add webhook"

### Testing

Test the integration:
```bash
python -m integrations.gitlab_integration format --issues 5
```

## 🔌 FastAPI Integration

### Update Your FastAPI App

In `webapp/app.py`, add the webhook router:

```python
from webapp.webhooks import router as webhook_router

# After creating the FastAPI app
app.include_router(webhook_router)
```

### Verify Endpoints

Check that webhook endpoints are registered:
```bash
curl https://your-server.com/webhook/health
```

Expected response:
```json
{
  "status": "healthy",
  "processed_events": 0,
  "github_configured": true,
  "gitlab_configured": true
}
```

## 🛡️ Security Best Practices

### 1. Token Security
- Never commit tokens to version control
- Use environment variables or secure secret management
- Rotate tokens regularly
- Use minimal required permissions

### 2. Webhook Security
- Always verify signatures (GitHub) or tokens (GitLab)
- Use HTTPS for webhook endpoints
- Implement rate limiting to prevent abuse

### 3. Network Security
- Whitelist GitHub/GitLab IP ranges if possible
- Use a reverse proxy (nginx) with rate limiting
- Monitor webhook endpoint logs

### 4. Code Security
- Run analysis in isolated environments
- Limit resource usage for analysis jobs
- Sanitize any user input before display

## 📊 Rate Limiting

The integration includes built-in rate limiting:
- Maximum 1 comment per repository per 5 minutes
- Prevents spam during rapid PR updates
- Updates existing comments instead of creating new ones

## 🔄 Idempotency

The webhook handlers are idempotent:
- Duplicate events are detected and skipped
- Event IDs are generated from PR/MR metadata
- Safe for webhook retries

## 📝 Comment Format

The bot posts structured comments including:
- Summary of issues by severity
- Top 10 most critical issues
- Actionable suggestions for each issue
- Links to detailed reports

Example comment structure:
```markdown
## 🔍 Code Quality Analysis Results

Found **15** issue(s) in this pull request.

### Summary by Severity
| Severity | Count |
|----------|-------|
| 🔴 Critical | 2 |
| 🟠 High | 5 |
| 🟡 Medium | 8 |

### Top 10 Issues
| Severity | File | Line | Issue | Suggestion |
|----------|------|------|-------|------------|
| 🔴 critical | auth.py | 45 | SQL injection risk | Sanitize input |
...
```

## 🚨 Monitoring

### Logs
Monitor application logs for webhook processing:
```bash
tail -f app.log | grep webhook
```

### Metrics to Track
- Webhook delivery success rate
- Analysis completion time
- Comment posting success rate
- Rate limit hits

## 🐛 Common Issues

### GitHub

**Issue**: Webhook delivers but no comment appears
- Check PR has a valid head repository
- Verify token has write permissions
- Check logs for clone/analysis errors

**Issue**: "Resource not accessible by integration"
- Token lacks required permissions
- PR is from a fork with restricted access

### GitLab

**Issue**: 404 errors when posting comments
- Check project ID is correct
- Verify token has API access
- Ensure MR exists and is accessible

**Issue**: Analysis runs but comment fails
- Check merge request permissions
- Verify bot user can comment on MRs

## 📚 Additional Resources

- [GitHub Webhooks Documentation](https://docs.github.com/en/developers/webhooks-and-events)
- [GitLab Webhooks Documentation](https://docs.gitlab.com/ee/user/project/integrations/webhooks.html)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)

## 🤝 Support

For issues or questions:
1. Check the application logs
2. Verify environment variables
3. Test with sample payloads
4. Open an issue in the repository