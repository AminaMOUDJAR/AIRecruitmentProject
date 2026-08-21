// TalentMatch AI - Enterprise Platform Controller
const API_BASE = "";

// State
let state = {
  activeRole: "recruiter",
  jobs: [],
  candidates: [],
  selectedJobId: null,
  loggedInCandidateId: localStorage.getItem("talentmatch_candidate_id") || "cand-001",
  activeMatches: []
};

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  try {
    await Promise.all([loadJobs(), loadCandidates()]);

    if (state.jobs.length > 0) {
      state.selectedJobId = state.jobs[0].id;
      renderJobSelector();
      renderActiveJobDetails();
      await fetchJobMatches(state.selectedJobId);
    }

    renderSeekerViewState();
  } catch (err) {
    console.error("Initialization error:", err);
    showToast("Error connecting to backend API");
  }
}

async function loadJobs() {
  const res = await fetch(`${API_BASE}/api/jobs`);
  state.jobs = await res.json();
}

async function loadCandidates() {
  const res = await fetch(`${API_BASE}/api/candidates`);
  state.candidates = await res.json();
}

// Navigation Role Switch
function switchRole(role) {
  state.activeRole = role;
  const tabRecruiter = document.getElementById("tabRecruiter");
  const tabSeeker = document.getElementById("tabSeeker");
  const recruiterView = document.getElementById("recruiterView");
  const seekerView = document.getElementById("seekerView");

  if (role === "recruiter") {
    tabRecruiter.classList.add("active");
    tabSeeker.classList.remove("active");
    recruiterView.classList.add("active");
    seekerView.classList.remove("active");
    if (state.selectedJobId) {
      fetchJobMatches(state.selectedJobId);
    }
  } else {
    tabSeeker.classList.add("active");
    tabRecruiter.classList.remove("active");
    seekerView.classList.add("active");
    recruiterView.classList.remove("active");
    renderSeekerViewState();
  }
}

// Recruiter Logic
function renderJobSelector() {
  const select = document.getElementById("recruiterJobSelect");
  select.innerHTML = state.jobs
    .map(
      (job) =>
        `<option value="${job.id}" ${
          job.id === state.selectedJobId ? "selected" : ""
        }>${escapeHtml(job.title)} &bull; ${escapeHtml(job.company)}</option>`
    )
    .join("");
}

function onRecruiterJobChange() {
  const select = document.getElementById("recruiterJobSelect");
  state.selectedJobId = select.value;
  renderActiveJobDetails();
  fetchJobMatches(state.selectedJobId);
}

function renderActiveJobDetails() {
  const job = state.jobs.find((j) => j.id === state.selectedJobId);
  if (!job) return;

  document.getElementById("activeJobTitle").textContent = job.title;
  document.getElementById("activeJobCompany").textContent = job.company;
  document.getElementById("activeJobLocation").textContent = job.location;
  document.getElementById("activeJobExp").textContent =
    job.experience_level || `${job.min_experience_years}+ Years`;
  document.getElementById("activeJobDescription").textContent = job.description;

  const reqContainer = document.getElementById("activeJobSkills");
  reqContainer.innerHTML = (job.required_skills || [])
    .map((s) => `<span class="tag tag-brand">${escapeHtml(s)}</span>`)
    .join("");

  const niceContainer = document.getElementById("activeJobNiceSkills");
  niceContainer.innerHTML = (job.nice_to_have_skills || [])
    .map((s) => `<span class="tag">${escapeHtml(s)}</span>`)
    .join("");
}

async function fetchJobMatches(jobId) {
  const listEl = document.getElementById("rankedCandidatesList");
  listEl.innerHTML = `
    <div class="loading-placeholder">
      <div class="spinner-border"></div>
      <div style="margin-top: 8px;">Calculating semantic vector scores...</div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/match/job/${jobId}`, {
      method: "POST"
    });
    const data = await res.json();
    state.activeMatches = data.matches || [];

    document.getElementById("candidateCountBadge").textContent = `${state.activeMatches.length} Evaluated`;
    renderRankedCandidates(state.activeMatches);
  } catch (err) {
    listEl.innerHTML = `<p style="color: var(--badge-danger-text); padding: 14px;">Error calculating candidate rankings.</p>`;
  }
}

function renderRankedCandidates(matches) {
  const listEl = document.getElementById("rankedCandidatesList");
  if (!matches || matches.length === 0) {
    listEl.innerHTML = `<p style="color: var(--text-muted); padding: 16px;">No candidates in database.</p>`;
    return;
  }

  listEl.innerHTML = matches
    .map((m) => {
      const cand = m.candidate;
      const score = m.match_score;
      let tierClass = "tier-high";
      if (score < 55) tierClass = "tier-low";
      else if (score < 75) tierClass = "tier-mid";

      const matchedSkillsHtml = (m.matched_skills || [])
        .map((s) => `<span class="tag tag-success">Matched: ${escapeHtml(s)}</span>`)
        .join("");

      const missingSkillsHtml = (m.missing_skills || [])
        .map((s) => `<span class="tag tag-warning">Gap: ${escapeHtml(s)}</span>`)
        .join("");

      const ragSnippet =
        m.rag_evidence && m.rag_evidence.length > 0
          ? m.rag_evidence[0].chunk_text
          : cand.bio;

      return `
      <div class="candidate-row-card">
        <div class="score-badge-box ${tierClass}">
          <div class="score-value">${score}%</div>
          <div class="score-text">Fit Score</div>
        </div>

        <div>
          <div class="candidate-name">${escapeHtml(cand.name)} <span style="font-weight: 400; font-size: 12px; color: var(--text-muted);">(${cand.years_experience} yrs exp)</span></div>
          <div class="candidate-headline">${escapeHtml(cand.title)} &bull; ${escapeHtml(cand.location || "Remote")}</div>
          <div class="candidate-summary-text">${escapeHtml(cand.bio || "")}</div>

          <div class="tag-group">
            ${matchedSkillsHtml}
            ${missingSkillsHtml}
          </div>

          ${
            ragSnippet
              ? `
            <div class="evidence-box">
              <div class="evidence-box-label">RAG Resume Evidence</div>
              <div>"${escapeHtml(ragSnippet.substring(0, 160))}..."</div>
            </div>
          `
              : ""
          }
        </div>

        <div>
          <button class="btn btn-secondary btn-sm" onclick="openCandidateDeepAnalysis('${cand.id}', '${state.selectedJobId}')">
            Evaluation Scorecard
          </button>
        </div>
      </div>
    `;
    })
    .join("");
}

// AI Evaluation Modal
async function openCandidateDeepAnalysis(candidateId, jobId) {
  openModal("aiAnalysisModal");
  const modalBody = document.getElementById("modalAnalysisBody");
  const candidate = state.candidates.find((c) => c.id === candidateId);
  const job = state.jobs.find((j) => j.id === jobId);

  document.getElementById("modalCandidateName").textContent = candidate
    ? `${candidate.name} — Evaluation Scorecard`
    : "Candidate Assessment";
  document.getElementById("modalJobTitleSub").textContent = job
    ? `Target Position: ${job.title} (${job.company})`
    : "Position Match Analysis";

  modalBody.innerHTML = `
    <div class="loading-placeholder">
      <div class="spinner-border"></div>
      <div style="margin-top: 8px;">Generating structured interview scorecard...</div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/ai/candidate-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_id: candidateId, job_id: jobId })
    });
    const analysis = await res.json();
    renderAnalysisModalContent(analysis);
  } catch (err) {
    modalBody.innerHTML = `<p style="color: var(--badge-danger-text);">Failed to load evaluation analysis.</p>`;
  }
}

function renderAnalysisModalContent(data) {
  const modalBody = document.getElementById("modalAnalysisBody");

  const strengthsHtml = (data.strengths || [])
    .map((s) => `<li style="margin-bottom: 4px; color: var(--text-body);">&#8226; ${escapeHtml(s)}</li>`)
    .join("");

  const risksHtml = (data.risks || [])
    .map((r) => `<li style="margin-bottom: 4px; color: var(--text-body);">&#8226; ${escapeHtml(r)}</li>`)
    .join("");

  const questionsHtml = (data.interview_questions || [])
    .map(
      (q) => `
      <div class="interview-q-card">
        <div class="interview-q-category">${escapeHtml(q.category)} &bull; ${escapeHtml(q.topic)}</div>
        <div class="interview-q-question">${escapeHtml(q.question)}</div>
        <div class="interview-q-notes">Assessment Objective: ${escapeHtml(q.rationale)}</div>
      </div>
    `
    )
    .join("");

  modalBody.innerHTML = `
    <!-- Executive Scorecard Summary -->
    <div class="scorecard-summary-card">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
        <span class="scorecard-title">Executive Candidate Summary</span>
        <span class="tag tag-brand">${data.match_score}% Semantic Fit</span>
      </div>
      <p style="font-size: 13px; color: var(--text-heading); line-height: 1.5; margin-bottom: 6px;">
        ${escapeHtml(data.executive_summary)}
      </p>
      <div style="font-size: 12px; font-weight: 600; color: var(--brand-primary);">
        Hiring Recommendation: <span style="font-weight: 400; color: var(--text-body);">${escapeHtml(data.recommendation)}</span>
      </div>
    </div>

    <!-- Strengths & Gaps Table -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px;">
      <div style="background: var(--badge-success-bg); border: 1px solid var(--badge-success-border); border-radius: var(--radius-sm); padding: 12px;">
        <div style="font-size: 12px; font-weight: 600; color: var(--badge-success-text); margin-bottom: 6px;">Key Competencies</div>
        <ul style="list-style: none; font-size: 12px;">${strengthsHtml}</ul>
      </div>

      <div style="background: var(--badge-warning-bg); border: 1px solid var(--badge-warning-border); border-radius: var(--radius-sm); padding: 12px;">
        <div style="font-size: 12px; font-weight: 600; color: var(--badge-warning-text); margin-bottom: 6px;">Identified Skill Gaps</div>
        <ul style="list-style: none; font-size: 12px;">${risksHtml}</ul>
      </div>
    </div>

    <!-- Structured Interview Questions -->
    <div>
      <div style="font-size: 13px; font-weight: 600; color: var(--text-heading); margin-bottom: 8px;">
        Suggested Technical Interview Questions
      </div>
      ${questionsHtml}
    </div>
  `;
}

// Candidate Portal State
function renderSeekerViewState() {
  const authSection = document.getElementById("seekerAuthSection");
  const dashboardSection = document.getElementById("seekerDashboardSection");

  const cand = state.candidates.find((c) => c.id === state.loggedInCandidateId);

  if (!cand) {
    authSection.style.display = "block";
    dashboardSection.style.display = "none";
    return;
  }

  authSection.style.display = "none";
  dashboardSection.style.display = "block";

  const initials = cand.name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .substring(0, 2)
    .toUpperCase();

  document.getElementById("seekerAvatar").textContent = initials || "CA";
  document.getElementById("seekerHeaderName").textContent = cand.name;
  document.getElementById("seekerHeaderEmail").textContent = cand.email;

  document.getElementById("seekerName").value = cand.name || "";
  document.getElementById("seekerTitle").value = cand.title || "";
  document.getElementById("seekerEmail").value = cand.email || "";
  document.getElementById("seekerSkills").value = (cand.skills || []).join(", ");
  document.getElementById("seekerBio").value = cand.bio || "";

  fetchSeekerMatchedJobs(cand.id);
  fetchSeekerTips(cand.id);
}

function handleCandidateLogin(event) {
  event.preventDefault();
  const emailInput = document.getElementById("loginEmail").value.trim().toLowerCase();
  const found = state.candidates.find((c) => c.email.toLowerCase() === emailInput);

  if (found) {
    state.loggedInCandidateId = found.id;
    localStorage.setItem("talentmatch_candidate_id", found.id);
    showToast(`Signed in as ${found.name}`);
    renderSeekerViewState();
  } else {
    showToast("Profile not found. You can upload a resume below to register.");
  }
}

function quickLoginCandidate(candId) {
  state.loggedInCandidateId = candId;
  localStorage.setItem("talentmatch_candidate_id", candId);
  const cand = state.candidates.find((c) => c.id === candId);
  showToast(`Signed in as ${cand ? cand.name : "Candidate"}`);
  renderSeekerViewState();
}

function candidateLogout() {
  state.loggedInCandidateId = null;
  localStorage.removeItem("talentmatch_candidate_id");
  showToast("Signed out");
  renderSeekerViewState();
}

async function saveSeekerProfile(event) {
  event.preventDefault();
  const cand = state.candidates.find((c) => c.id === state.loggedInCandidateId) || {};

  const skillsArray = document
    .getElementById("seekerSkills")
    .value.split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const payload = {
    ...cand,
    id: state.loggedInCandidateId,
    name: document.getElementById("seekerName").value,
    title: document.getElementById("seekerTitle").value,
    email: document.getElementById("seekerEmail").value,
    skills: skillsArray,
    bio: document.getElementById("seekerBio").value,
    experience: cand.experience || [],
    education: cand.education || []
  };

  try {
    const res = await fetch(`${API_BASE}/api/candidates`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    await res.json();
    showToast("Profile saved successfully.");

    await loadCandidates();
    renderSeekerViewState();
    if (state.selectedJobId) {
      fetchJobMatches(state.selectedJobId);
    }
  } catch (err) {
    showToast("Failed to save profile updates.");
  }
}

async function fetchSeekerMatchedJobs(candidateId) {
  const listEl = document.getElementById("seekerMatchedJobsList");
  listEl.innerHTML = `
    <div class="loading-placeholder">
      <div class="spinner-border"></div>
      <div style="margin-top: 6px;">Ranking requisitions...</div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/match/candidate/${candidateId}`, {
      method: "POST"
    });
    const data = await res.json();
    const matchedJobs = data.matched_jobs || [];

    document.getElementById("seekerJobMatchCount").textContent = `${matchedJobs.length} Positions`;

    if (matchedJobs.length === 0) {
      listEl.innerHTML = `<p style="color: var(--text-muted); padding: 14px;">No positions available.</p>`;
      return;
    }

    listEl.innerHTML = matchedJobs
      .map((item) => {
        const job = item.job;
        const score = item.match_score;
        let tierClass = "tier-high";
        if (score < 55) tierClass = "tier-low";
        else if (score < 75) tierClass = "tier-mid";

        const matchedSkillsHtml = (item.matched_skills || [])
          .map((s) => `<span class="tag tag-success">Matched: ${escapeHtml(s)}</span>`)
          .join("");

        const missingSkillsHtml = (item.missing_skills || [])
          .map((s) => `<span class="tag tag-warning">Gap: ${escapeHtml(s)}</span>`)
          .join("");

        return `
        <div class="candidate-row-card">
          <div class="score-badge-box ${tierClass}">
            <div class="score-value">${score}%</div>
            <div class="score-text">Match</div>
          </div>

          <div>
            <div class="candidate-name">${escapeHtml(job.title)}</div>
            <div class="candidate-headline">${escapeHtml(job.company)} &bull; ${escapeHtml(job.location)}</div>
            <div class="candidate-summary-text">${escapeHtml(job.description)}</div>

            <div class="tag-group">
              ${matchedSkillsHtml}
              ${missingSkillsHtml}
            </div>
          </div>

          <div>
            <button class="btn btn-primary btn-sm" onclick="showToast('Application submitted for ${escapeHtml(job.title)}')">
              Apply Now
            </button>
          </div>
        </div>
      `;
      })
      .join("");
  } catch (err) {
    listEl.innerHTML = `<p style="color: var(--badge-danger-text);">Failed to load matched jobs.</p>`;
  }
}

async function fetchSeekerTips(candidateId) {
  const container = document.getElementById("seekerTipsContainer");
  container.innerHTML = `
    <div class="loading-placeholder">
      <div class="spinner-border"></div>
      <div style="margin-top: 4px;">Analyzing resume...</div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/ai/resume-tips/${candidateId}`, {
      method: "POST"
    });
    const data = await res.json();
    const tips = data.tips || [];

    if (tips.length === 0) {
      container.innerHTML = `<p style="color: var(--text-muted); font-size: 13px;">No optimization issues detected.</p>`;
      return;
    }

    container.innerHTML = tips
      .map(
        (t) => `
      <div class="tip-item">
        <div class="tip-item-title">${escapeHtml(t.title)}</div>
        <div class="tip-item-desc">${escapeHtml(t.description)}</div>
      </div>
    `
      )
      .join("");
  } catch (err) {
    container.innerHTML = `<p style="color: var(--text-muted);">Tips unavailable.</p>`;
  }
}

function handleCvFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    uploadCvFile(file);
  }
}

async function uploadCvFile(file) {
  showToast(`Parsing ${file.name}...`);
  const formData = new FormData();
  formData.append("file", file);
  formData.append("candidate_name", file.name.replace(/\.[^/.]+$/, ""));

  try {
    const res = await fetch(`${API_BASE}/api/candidates/upload-cv`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    showToast("Resume parsed and profile imported.");

    await loadCandidates();
    state.loggedInCandidateId = data.candidate.id;
    localStorage.setItem("talentmatch_candidate_id", data.candidate.id);
    renderSeekerViewState();
  } catch (err) {
    showToast("Failed to parse resume document.");
  }
}

async function handlePostNewJob(event) {
  event.preventDefault();

  const reqSkills = document
    .getElementById("postJobRequiredSkills")
    .value.split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const niceSkills = document
    .getElementById("postJobNiceSkills")
    .value.split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const payload = {
    title: document.getElementById("postJobTitle").value,
    company: document.getElementById("postJobCompany").value,
    location: document.getElementById("postJobLocation").value,
    type: "Full-Time",
    experience_level: `${document.getElementById("postJobMinExp").value}+ Years`,
    description: document.getElementById("postJobDescription").value,
    required_skills: reqSkills,
    nice_to_have_skills: niceSkills,
    min_experience_years: parseInt(document.getElementById("postJobMinExp").value) || 3
  };

  try {
    const res = await fetch(`${API_BASE}/api/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    showToast("Job requisition published.");

    closeModal("postJobModal");
    document.getElementById("newJobForm").reset();

    await loadJobs();
    state.selectedJobId = result.job.id;
    renderJobSelector();
    renderActiveJobDetails();
    await fetchJobMatches(state.selectedJobId);
  } catch (err) {
    showToast("Failed to post job requisition.");
  }
}

async function resetDemoData() {
  if (!confirm("Reset all jobs and candidates to original dataset?")) return;
  try {
    await fetch(`${API_BASE}/api/reset-data`, { method: "POST" });
    showToast("Dataset reset to original state.");
    await loadJobs();
    await loadCandidates();
    state.loggedInCandidateId = "cand-001";
    localStorage.setItem("talentmatch_candidate_id", "cand-001");
    renderJobSelector();
    renderActiveJobDetails();
    fetchJobMatches(state.selectedJobId);
    renderSeekerViewState();
  } catch (err) {
    showToast("Reset failed.");
  }
}

function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.add("active");
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.remove("active");
}

window.addEventListener("click", (e) => {
  if (e.target.classList.contains("modal-backdrop")) {
    e.target.classList.remove("active");
  }
});

function showToast(message) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast-message`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 250);
  }, 2800);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
