package com.nedama.signal.gateway83;

import com.inductiveautomation.ignition.common.licensing.LicenseState;
import com.inductiveautomation.ignition.common.script.ScriptManager;
import com.inductiveautomation.ignition.gateway.model.AbstractGatewayModuleHook;
import com.inductiveautomation.ignition.gateway.model.GatewayContext;
import com.nedama.signal.common83.SignalRegistration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Gateway scope: gateway scripts, tag event scripts, and Perspective scripts and bindings. */
public class SignalGatewayHook extends AbstractGatewayModuleHook {

    private static final Logger LOG = LoggerFactory.getLogger("SignalProcessing");

    @Override
    public void setup(GatewayContext context) {
        // Nothing to set up: the module only adds functions.
    }

    @Override
    public void startup(LicenseState activationState) {
        LOG.info("Signal Processing started: system.signal functions are available.");
    }

    @Override
    public void shutdown() {
        // Nothing to release.
    }

    @Override
    public void initializeScriptManager(ScriptManager manager) {
        super.initializeScriptManager(manager);
        SignalRegistration.registerScripts(manager);
    }

    @Override
    public boolean isFreeModule() {
        return true;
    }

    @Override
    public boolean isMakerEditionCompatible() {
        return true;
    }
}
